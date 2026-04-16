#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
correcao.py

Correção híbrida de questões dinâmicas com alta robustez:

- correcao: valida a solução do aluno com casos de teste
- modificacao: valida com testes + checagem de requisitos do enunciado
- previsao: executa o código dado e compara a saída com a resposta do aluno
- justificativa: correção semântica com LLM
- descritiva: correção semântica com LLM

Entrada padrão:
    conteudo/perguntasGeradas.txt

Saída:
    conteudo/correcao.txt

Compatível com:
- arquivo JSON
- arquivo texto em blocos com rótulos como:
  TIPO:, ENUNCIADO:, RESPOSTA DO ALUNO:, CODIGO:, TESTES:, etc.

Recomendação:
- usar um servidor OpenAI-compatible local, como LLM Studio, com:
  LLM_BASE_URL=http://localhost:1234/v1
  LLM_MODEL=qwen/qwen3-vl-4b
"""

from __future__ import annotations

import ast
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import traceback
import unicodedata
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =========================
# Configuração
# =========================

ARQUIVO_ENTRADA = Path(os.getenv("ARQUIVO_ENTRADA", "conteudo/perguntasGeradas.txt"))
ARQUIVO_SAIDA = Path(os.getenv("ARQUIVO_SAIDA", "conteudo/correcao.txt"))

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3-vl-4b")
USAR_LLM = os.getenv("USAR_LLM", "1").strip() not in {"0", "false", "False", "no", "NO"}
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "90"))

# Pontos de corte para aceitar comparações textuais aproximadas
LIMIAR_EXATO = 0.99
LIMIAR_APROX = 0.90

# Número alvo de testes por questão de código
TESTES_ALVO = 6


# =========================
# Modelos de dados
# =========================

@dataclass
class Questao:
    idx: int
    tipo: str = ""
    enunciado: str = ""
    resposta_aluno: str = ""
    resposta_referencia: str = ""
    codigo: str = ""
    entrada: str = ""
    saida: str = ""
    testes: List[Dict[str, str]] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Resultado:
    idx: int
    tipo: str
    nota: float
    status: str
    feedback: str
    detalhes: List[str] = field(default_factory=list)
    testes_executados: List[Dict[str, Any]] = field(default_factory=list)
    saida_correta: str = ""


# =========================
# Utilidades de texto
# =========================

def sem_acentos(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto)
    return "".join(ch for ch in texto if unicodedata.category(ch) != "Mn")


def normalizar_label(label: str) -> str:
    label = sem_acentos(label).lower().strip()
    label = re.sub(r"[^a-z0-9]+", " ", label)
    return re.sub(r"\s+", " ", label).strip()


def normalizar_tipo(tipo: str) -> str:
    t = normalizar_label(tipo)
    mapa = {
        "correcao": "correcao",
        "corrigir": "correcao",
        "correcao de codigo": "correcao",
        "justificativa": "justificativa",
        "justificar": "justificativa",
        "descritiva": "descritiva",
        "descrever": "descritiva",
        "modificacao": "modificacao",
        "modificar": "modificacao",
        "alteracao": "modificacao",
        "alterar": "modificacao",
        "previsao": "previsao",
        "prever": "previsao",
    }
    return mapa.get(t, t or "")


def normalizar_texto(texto: str) -> str:
    if texto is None:
        return ""
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    linhas = [ln.rstrip() for ln in texto.split("\n")]
    while linhas and not linhas[0].strip():
        linhas.pop(0)
    while linhas and not linhas[-1].strip():
        linhas.pop()
    return "\n".join(linhas).strip()


def compactar_texto(texto: str) -> str:
    texto = normalizar_texto(texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def tokenizar(texto: str) -> List[str]:
    texto = sem_acentos(texto.lower())
    return re.findall(r"[a-z0-9_]+", texto)


def comparar_textos(a: str, b: str) -> float:
    """
    Retorna uma similaridade entre 0 e 1.
    """
    a0 = compactar_texto(a)
    b0 = compactar_texto(b)

    if a0 == b0:
        return 1.0

    a1 = a0.replace(" ", "")
    b1 = b0.replace(" ", "")
    if a1 == b1 and a1:
        return 0.98

    ratio = difflib.SequenceMatcher(None, a0.lower(), b0.lower()).ratio()
    return ratio


def extrair_codigo(texto: str) -> str:
    """
    Extrai código entre cercas markdown. Se não houver cercas,
    retorna o texto bruto.
    """
    if not texto:
        return ""

    blocos = re.findall(
        r"```(?:python|py|c|cpp|java|javascript|txt|text)?\s*\n(.*?)```",
        texto,
        flags=re.S | re.I,
    )
    if blocos:
        return "\n\n".join(bloco.strip() for bloco in blocos if bloco.strip())

    return texto.strip()


def extrair_json(texto: str) -> Optional[Any]:
    if not texto:
        return None

    texto = texto.strip()

    # remove cercas markdown se existirem
    if texto.startswith("```"):
        texto = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", texto)
        texto = re.sub(r"\n?```$", "", texto).strip()

    try:
        return json.loads(texto)
    except Exception:
        pass

    candidatos = []
    m = re.search(r"\{.*\}", texto, flags=re.S)
    if m:
        candidatos.append(m.group(0))
    m = re.search(r"\[.*\]", texto, flags=re.S)
    if m:
        candidatos.append(m.group(0))

    for c in candidatos:
        try:
            return json.loads(c)
        except Exception:
            continue

    return None


def exige_saida_no_enunciado(texto: str) -> bool:
    """
    Detecta se o enunciado realmente pede saída na tela / retorno.
    Isso evita penalizar questões que só pedem modificação estrutural do código.
    """
    t = sem_acentos((texto or "").lower())
    padroes = [
        r"\bprint(?:e|ar|e)?\b",
        r"\bmostre\b",
        r"\bimprima\b",
        r"\bexiba\b",
        r"\bsaida\b",
        r"\bretorne\b",
        r"\breturn\b",
        r"\bdeve retornar\b",
        r"\bmostrar o resultado\b",
    ]
    return any(re.search(p, t) for p in padroes)


# =========================
# Leitura e parsing
# =========================

ALIASES_CAMPOS = {
    "tipo": "tipo",
    "enunciado": "enunciado",
    "questao": "enunciado",
    "questão": "enunciado",
    "pergunta": "enunciado",
    "texto": "enunciado",
    "resposta do aluno": "resposta_aluno",
    "resposta aluno": "resposta_aluno",
    "resposta": "resposta_aluno",
    "answer": "resposta_aluno",
    "solucao do aluno": "resposta_aluno",
    "solução do aluno": "resposta_aluno",
    "resposta esperada": "resposta_referencia",
    "resposta modelo": "resposta_referencia",
    "modelo": "resposta_referencia",
    "gabarito": "resposta_referencia",
    "codigo": "codigo",
    "código": "codigo",
    "programa": "codigo",
    "trecho de codigo": "codigo",
    "trecho de código": "codigo",
    "stdin": "entrada",
    "input": "entrada",
    "entrada": "entrada",
    "saida": "saida",
    "saída": "saida",
    "output": "saida",
    "testes": "testes",
    "casos de teste": "testes",
}


def carregar_arquivo_texto(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def split_exercicios(texto: str) -> List[str]:
    """
    Tenta separar blocos que comecem com linhas do tipo:
    ========== EXERCÍCIO 1 ==========
    """
    padrao = re.compile(r"(?m)^\s*=+\s*EXERC[IÍ]CIO\s*\d+.*=+\s*$")
    matches = list(padrao.finditer(texto))
    if not matches:
        return [texto.strip()] if texto.strip() else []

    blocos = []
    for i, m in enumerate(matches):
        inicio = m.end()
        fim = matches[i + 1].start() if i + 1 < len(matches) else len(texto)
        bloco = texto[inicio:fim].strip()
        if bloco:
            blocos.append(bloco)
    return blocos if blocos else [texto.strip()]


def parse_tests_field(texto: str) -> List[Dict[str, str]]:
    """
    Suporta:
    - JSON list
    - linhas no formato: entrada => saida
    - blocos com entrada/saida
    """
    texto = normalizar_texto(texto)
    if not texto:
        return []

    if texto.startswith("["):
        try:
            obj = json.loads(texto)
            if isinstance(obj, list):
                testes = []
                for item in obj:
                    if isinstance(item, dict):
                        testes.append(
                            {
                                "entrada": str(item.get("entrada", item.get("input", ""))),
                                "saida": str(item.get("saida", item.get("output", ""))),
                                "obs": str(item.get("obs", item.get("descricao", ""))),
                            }
                        )
                return [t for t in testes if t["entrada"] or t["saida"]]
        except Exception:
            pass

    testes = []
    for linha in texto.splitlines():
        linha = linha.strip()
        if not linha:
            continue

        if "=>" in linha:
            esquerda, direita = linha.split("=>", 1)
            testes.append(
                {
                    "entrada": esquerda.strip(),
                    "saida": direita.strip(),
                    "obs": "",
                }
            )
        elif "|" in linha:
            esquerda, direita = linha.split("|", 1)
            testes.append(
                {
                    "entrada": esquerda.strip(),
                    "saida": direita.strip(),
                    "obs": "",
                }
            )

    return testes


def inferir_tipo(texto: str) -> str:
    t = sem_acentos((texto or "").lower())

    # Detecta rótulos explícitos no texto, inclusive em formatos como:
    # [MODIFICACAO], [CORRECAO], [JUSTIFICATIVA], [DESCRITIVA], [PREVISAO]
    if re.search(r"\[\s*(modificacao|alteracao)\s*\]", t):
        return "modificacao"
    if re.search(r"\[\s*(correcao|corrigir|corrija|conserte|erro)\s*\]", t):
        return "correcao"
    if re.search(r"\[\s*(previsao|prever)\s*\]", t):
        return "previsao"
    if re.search(r"\[\s*(justificativa|justificar|explique)\s*\]", t):
        return "justificativa"
    if re.search(r"\[\s*(descritiva|descrever)\s*\]", t):
        return "descritiva"

    if any(p in t for p in [
        "qual sera a saida",
        "o que imprime",
        "o que sera impresso",
        "preveja a saida",
        "previsao",
        "resultado da execucao",
        "saida do programa",
    ]):
        return "previsao"

    if any(p in t for p in [
        "corrija",
        "corrigir",
        "conserte",
        "erro",
        "bug",
        "falha no codigo",
        "correcao",
    ]):
        return "correcao"

    if any(p in t for p in [
        "modifique",
        "modificar",
        "altere",
        "alterar",
        "adapte",
        "reescreva",
        "adicione",
        "incluir",
        "remova",
        "modificacao",
        "alteracao",
    ]):
        return "modificacao"

    if any(p in t for p in [
        "justifique",
        "justificativa",
        "por que",
        "explique por que",
    ]):
        return "justificativa"

    if any(p in t for p in [
        "descreva",
        "descritiva",
        "resuma",
        "explique",
    ]):
        return "descritiva"

    return ""


def parse_block(block: str, idx: int) -> Questao:
    data: Dict[str, str] = {}
    current_field: Optional[str] = None
    buffer: List[str] = []

    def flush() -> None:
        nonlocal buffer, current_field
        if current_field is not None:
            conteudo = "\n".join(buffer).strip()
            if conteudo:
                if current_field in data and data[current_field]:
                    data[current_field] += "\n" + conteudo
                else:
                    data[current_field] = conteudo
        buffer = []

    for linha in block.splitlines():
        m = re.match(r"^\s*([A-Za-zÀ-ÿ0-9 _/()\-]{2,80})\s*:\s*(.*)$", linha)
        if m:
            label = normalizar_label(m.group(1))
            if label in ALIASES_CAMPOS:
                flush()
                current_field = ALIASES_CAMPOS[label]
                primeiro_valor = m.group(2).strip()
                buffer = [primeiro_valor] if primeiro_valor else []
                continue

        if current_field is None:
            data["enunciado"] = data.get("enunciado", "") + linha + "\n"
        else:
            buffer.append(linha)

    flush()

    enunciado = normalizar_texto(data.get("enunciado", ""))
    resposta_aluno = normalizar_texto(data.get("resposta_aluno", ""))
    resposta_referencia = normalizar_texto(data.get("resposta_referencia", ""))
    codigo = normalizar_texto(data.get("codigo", ""))

    if not codigo:
        codigo = extrair_codigo(enunciado)
        if not codigo and resposta_aluno:
            codigo = extrair_codigo(resposta_aluno)

    tipo = normalizar_tipo(data.get("tipo", "")) or inferir_tipo(enunciado)

    entrada = normalizar_texto(data.get("entrada", ""))
    saida = normalizar_texto(data.get("saida", ""))
    testes = parse_tests_field(data.get("testes", ""))

    extras = {
        k: v
        for k, v in data.items()
        if k not in {
            "tipo",
            "enunciado",
            "resposta_aluno",
            "resposta_referencia",
            "codigo",
            "entrada",
            "saida",
            "testes",
        }
    }

    return Questao(
        idx=idx,
        tipo=tipo,
        enunciado=enunciado,
        resposta_aluno=resposta_aluno,
        resposta_referencia=resposta_referencia,
        codigo=codigo,
        entrada=entrada,
        saida=saida,
        testes=testes,
        extras=extras,
    )


def carregar_questoes(path: Path) -> List[Questao]:
    texto = carregar_arquivo_texto(path).strip()
    if not texto:
        return []

    # Tenta JSON primeiro
    if texto[0] in "{[":
        try:
            obj = json.loads(texto)
            if isinstance(obj, dict):
                itens = obj.get("perguntas") or obj.get("questoes") or obj.get("questions") or []
            elif isinstance(obj, list):
                itens = obj
            else:
                itens = []

            questoes = []
            for i, item in enumerate(itens, start=1):
                if not isinstance(item, dict):
                    continue

                q = Questao(
                    idx=int(item.get("id", i)),
                    tipo=normalizar_tipo(str(item.get("tipo", "")))
                    or inferir_tipo(str(item.get("enunciado", item.get("pergunta", "")))),
                    enunciado=normalizar_texto(str(item.get("enunciado", item.get("pergunta", "")))),
                    resposta_aluno=normalizar_texto(str(item.get("resposta_aluno", item.get("resposta", "")))),
                    resposta_referencia=normalizar_texto(str(item.get("resposta_referencia", item.get("gabarito", "")))),
                    codigo=normalizar_texto(str(item.get("codigo", item.get("programa", "")))),
                    entrada=normalizar_texto(str(item.get("entrada", item.get("input", "")))),
                    saida=normalizar_texto(str(item.get("saida", item.get("output", "")))),
                    testes=parse_tests_field(
                        json.dumps(item.get("testes", []), ensure_ascii=False)
                        if item.get("testes") is not None
                        else ""
                    ),
                    extras={
                        k: v
                        for k, v in item.items()
                        if k
                        not in {
                            "id",
                            "tipo",
                            "enunciado",
                            "pergunta",
                            "resposta_aluno",
                            "resposta",
                            "resposta_referencia",
                            "gabarito",
                            "codigo",
                            "programa",
                            "entrada",
                            "input",
                            "saida",
                            "output",
                            "testes",
                        }
                    },
                )
                if not q.codigo:
                    q.codigo = extrair_codigo(q.enunciado or q.resposta_aluno)
                questoes.append(q)

            if questoes:
                return questoes
        except Exception:
            pass

    blocos = split_exercicios(texto)
    if len(blocos) == 1:
        return [parse_block(blocos[0], 1)]

    return [parse_block(bloco, i + 1) for i, bloco in enumerate(blocos)]


# =========================
# LLM
# =========================

def chamar_llm(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 1200,
) -> Optional[str]:
    if not USAR_LLM:
        return None

    url = LLM_BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    dados = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=dados,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        obj = json.loads(raw)
        return obj["choices"][0]["message"]["content"]
    except Exception:
        return None


def chamar_llm_json(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 1200,
) -> Optional[Any]:
    texto = chamar_llm(messages, temperature=temperature, max_tokens=max_tokens)
    if texto is None:
        return None
    return extrair_json(texto)


# =========================
# Execução de código
# =========================

def verificar_sintaxe_python(codigo: str) -> Tuple[bool, str]:
    try:
        ast.parse(codigo)
        return True, ""
    except SyntaxError as e:
        linha = f" linha {e.lineno}" if e.lineno else ""
        return False, f"{e.msg}{linha}"
    except Exception as e:
        return False, str(e)


def executar_codigo_python(codigo: str, entrada: str, timeout: int = 3) -> Dict[str, Any]:
    """
    Executa código Python em processo separado usando o interpretador atual.
    Retorna: stdout, stderr, retorno, timeout, erro_execucao
    """
    codigo = normalizar_texto(codigo)
    entrada = entrada if entrada is not None else ""

    resultado = {
        "stdout": "",
        "stderr": "",
        "returncode": None,
        "timeout": False,
        "erro_execucao": "",
    }

    if not codigo.strip():
        resultado["erro_execucao"] = "Código vazio"
        return resultado

    with tempfile.TemporaryDirectory() as td:
        caminho = Path(td) / "resposta_aluno.py"
        caminho.write_text(codigo, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(caminho)],
                input=entrada,
                text=True,
                capture_output=True,
                timeout=timeout,
                cwd=td,
            )
            resultado["stdout"] = proc.stdout or ""
            resultado["stderr"] = proc.stderr or ""
            resultado["returncode"] = proc.returncode
        except subprocess.TimeoutExpired as e:
            resultado["timeout"] = True
            resultado["stdout"] = e.stdout or ""
            resultado["stderr"] = e.stderr or ""
            resultado["erro_execucao"] = "Timeout"
        except Exception as e:
            resultado["erro_execucao"] = str(e)

    return resultado


def executar_codigo_python_sem_entrada(codigo: str, timeout: int = 3) -> Dict[str, Any]:
    return executar_codigo_python(codigo, "", timeout=timeout)


# =========================
# Testes
# =========================

def deduplicar_testes(testes: List[Dict[str, str]]) -> List[Dict[str, str]]:
    vistos = set()
    saida = []
    for t in testes:
        entrada = normalizar_texto(str(t.get("entrada", "")))
        chave = entrada
        if chave in vistos:
            continue
        vistos.add(chave)
        saida.append(
            {
                "entrada": entrada,
                "saida": normalizar_texto(str(t.get("saida", ""))),
                "obs": normalizar_texto(str(t.get("obs", ""))),
            }
        )
    return saida


def obter_testes_explicitos(q: Questao) -> List[Dict[str, str]]:
    testes = []
    if q.entrada or q.saida:
        testes.append({"entrada": q.entrada, "saida": q.saida, "obs": "Caso explícito do enunciado"})
    if q.testes:
        testes.extend(q.testes)
    return deduplicar_testes(testes)


def gerar_testes_com_llm(q: Questao, quantidade: int = TESTES_ALVO) -> List[Dict[str, str]]:
    if not USAR_LLM:
        return []

    schema = {
        "testes": [
            {
                "entrada": "1 2\n",
                "saida": "3\n",
                "obs": "caso simples",
            }
        ]
    }

    prompt = f"""
Você é um gerador rigoroso de casos de teste para questões de programação.

Tarefa:
- criar entre {max(4, quantidade)} e {quantidade + 2} casos de teste
- cobrir caso normal, bordas, entradas mínimas, entradas máximas e cenários problemáticos
- produzir saídas exatas
- não inventar comportamento fora do enunciado

Tipo da questão: {q.tipo}

Enunciado:
{q.enunciado}

Código de referência ou contexto adicional, se existir:
{q.codigo or "(não há)"}

Saída esperada, se existir:
{q.resposta_referencia or "(não há)"}

Retorne APENAS JSON válido neste formato:

{json.dumps(schema, ensure_ascii=False, indent=2)}

Regras:
- "entrada" deve ser exatamente o texto que o programa receberia no stdin
- "saida" deve ser exatamente o texto que o programa deveria imprimir
- use \\n quando necessário
- não escreva explicações fora do JSON
"""

    obj = chamar_llm_json(
        [
            {"role": "system", "content": "Você gera JSON válido e confiável para correção automática."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1500,
    )

    testes = []
    if isinstance(obj, dict):
        bruto = obj.get("testes", [])
        if isinstance(bruto, list):
            for item in bruto:
                if isinstance(item, dict):
                    testes.append(
                        {
                            "entrada": normalizar_texto(str(item.get("entrada", item.get("input", "")))),
                            "saida": normalizar_texto(str(item.get("saida", item.get("output", "")))),
                            "obs": normalizar_texto(str(item.get("obs", item.get("descricao", "")))),
                        }
                    )

    return deduplicar_testes(testes)[:quantidade]


def obter_testes(q: Questao) -> List[Dict[str, str]]:
    testes = obter_testes_explicitos(q)
    if len(testes) >= TESTES_ALVO:
        return testes[:TESTES_ALVO]

    if len(testes) < TESTES_ALVO:
        gerados = gerar_testes_com_llm(q, quantidade=TESTES_ALVO)
        testes.extend(gerados)

    testes = deduplicar_testes(testes)
    return testes[: max(TESTES_ALVO, len(testes))]


# =========================
# Correção por tipo
# =========================

def avaliar_previsao(q: Questao) -> Resultado:
    codigo_base = q.codigo or extrair_codigo(q.enunciado)
    if not codigo_base:
        return Resultado(
            idx=q.idx,
            tipo=q.tipo,
            nota=0.0,
            status="falha",
            feedback="Não foi possível localizar o código da questão para calcular a saída.",
            detalhes=["Faltou o trecho de código necessário para a previsão."],
        )

    entrada = q.entrada or ""
    execucao = executar_codigo_python(codigo_base, entrada, timeout=3)
    saida_correta = normalizar_texto(execucao["stdout"])

    resposta = q.resposta_aluno or ""
    sim = comparar_textos(resposta, saida_correta)

    if execucao["timeout"]:
        status = "parcial"
        feedback = "O código-base entrou em timeout durante a execução."
        nota = round(min(4.0, sim * 10), 2)
    elif execucao["erro_execucao"]:
        status = "parcial"
        feedback = f"Erro ao executar o código-base: {execucao['erro_execucao']}"
        nota = round(min(5.0, sim * 10), 2)
    elif sim >= LIMIAR_EXATO:
        status = "ok"
        feedback = "Resposta correta."
        nota = 10.0
    elif sim >= LIMIAR_APROX:
        status = "ok"
        feedback = "Resposta muito próxima da saída correta."
        nota = round(9.0 + (sim - LIMIAR_APROX) * 5, 2)
    else:
        status = "erro"
        feedback = "Resposta incorreta para a saída esperada."
        nota = round(sim * 10, 2)

    return Resultado(
        idx=q.idx,
        tipo=q.tipo,
        nota=max(0.0, min(10.0, nota)),
        status=status,
        feedback=feedback,
        detalhes=[
            "Saída correta calculada a partir do código-base.",
            f"Similaridade com a resposta do aluno: {sim:.3f}",
        ],
        testes_executados=[
            {
                "entrada": entrada,
                "saida_esperada": saida_correta,
                "saida_obtida": execucao["stdout"],
                "erro_execucao": execucao["erro_execucao"],
            }
        ],
        saida_correta=saida_correta,
    )


def avaliar_texto_heuristico(q: Questao) -> Resultado:
    resposta = normalizar_texto(q.resposta_aluno)
    if not resposta:
        return Resultado(
            idx=q.idx,
            tipo=q.tipo,
            nota=0.0,
            status="erro",
            feedback="Resposta vazia.",
            detalhes=["Sem resposta para avaliar."],
        )

    enunciado = normalizar_texto(q.enunciado)
    tokens_enunciado = set(tokenizar(enunciado))
    tokens_resposta = set(tokenizar(resposta))

    if not tokens_enunciado:
        return Resultado(
            idx=q.idx,
            tipo=q.tipo,
            nota=5.0,
            status="parcial",
            feedback="Não foi possível usar critérios semânticos completos; correção parcial por fallback.",
            detalhes=["Enunciado muito curto ou ausente."],
        )

    stop = {
        "a",
        "o",
        "e",
        "de",
        "do",
        "da",
        "dos",
        "das",
        "um",
        "uma",
        "uns",
        "umas",
        "que",
        "para",
        "por",
        "com",
        "sem",
        "em",
        "no",
        "na",
        "nos",
        "nas",
        "ao",
        "aos",
        "as",
        "os",
        "se",
        "nao",
        "não",
        "como",
        "porque",
        "qual",
        "quais",
        "ser",
        "sera",
        "será",
        "foi",
        "sao",
        "são",
        "isso",
        "isto",
    }

    tokens_enunciado = {t for t in tokens_enunciado if t not in stop and len(t) > 2}
    tokens_resposta = {t for t in tokens_resposta if t not in stop and len(t) > 2}

    overlap = len(tokens_enunciado & tokens_resposta) / max(1, len(tokens_enunciado))
    tamanho = min(1.0, len(tokens_resposta) / max(8, len(tokens_enunciado)))
    similaridade = comparar_textos(enunciado, resposta)

    nota = (0.45 * overlap + 0.20 * tamanho + 0.35 * similaridade) * 10
    nota = max(0.0, min(10.0, nota))

    if nota >= 8.5:
        status = "ok"
        feedback = "Resposta boa e coerente com o enunciado."
    elif nota >= 6.0:
        status = "parcial"
        feedback = "Resposta parcialmente correta, mas ainda pode ser mais precisa."
    else:
        status = "erro"
        feedback = "Resposta fraca em relação ao pedido do enunciado."

    return Resultado(
        idx=q.idx,
        tipo=q.tipo,
        nota=round(nota, 2),
        status=status,
        feedback=feedback,
        detalhes=[
            f"Sobreposição de termos: {overlap:.3f}",
            f"Fator de tamanho: {tamanho:.3f}",
            f"Similaridade textual: {similaridade:.3f}",
        ],
    )


def avaliar_texto_llm(q: Questao) -> Resultado:
    if not USAR_LLM:
        return avaliar_texto_heuristico(q)

    prompt = f"""
Você é um corretor rigoroso, mas justo.

Tipo da questão: {q.tipo}

Enunciado:
{q.enunciado}

Resposta do aluno:
{q.resposta_aluno or "(vazia)"}

Regras:
- avalie aderência ao pedido do enunciado
- considere correção conceitual, completude, clareza e objetividade
- penalize fuga do tema
- seja consistente com notas de 0 a 10
- NÃO use nota apenas 0 ou 10 se houver margem de parcialidade
- não invente informações que não estejam na resposta

Retorne APENAS JSON válido com este formato:

{{
  "nota": 0,
  "status": "ok|parcial|erro",
  "feedback": "texto curto e claro",
  "acertos": ["..."],
  "melhorias": ["..."]
}}

Use nota inteira ou decimal, mas sempre entre 0 e 10.
"""
    obj = chamar_llm_json(
        [
            {"role": "system", "content": "Você devolve JSON válido e faz correção de respostas textuais."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.15,
        max_tokens=1200,
    )

    if isinstance(obj, dict):
        try:
            nota = float(obj.get("nota", 0))
        except Exception:
            nota = 0.0

        status = str(obj.get("status", "parcial")).strip().lower()
        if status not in {"ok", "parcial", "erro"}:
            status = "parcial"

        acertos = obj.get("acertos", [])
        melhorias = obj.get("melhorias", [])
        if not isinstance(acertos, list):
            acertos = [str(acertos)]
        if not isinstance(melhorias, list):
            melhorias = [str(melhorias)]

        detalhes = []
        if acertos:
            detalhes.append("Acertos: " + "; ".join(str(x) for x in acertos[:5]))
        if melhorias:
            detalhes.append("Melhorias: " + "; ".join(str(x) for x in melhorias[:5]))

        return Resultado(
            idx=q.idx,
            tipo=q.tipo,
            nota=max(0.0, min(10.0, round(nota, 2))),
            status=status,
            feedback=str(obj.get("feedback", "")).strip() or "Correção sem feedback detalhado.",
            detalhes=detalhes,
        )

    return avaliar_texto_heuristico(q)


def avaliar_codigo_por_testes(q: Questao, codigo_aluno: str, testes: List[Dict[str, str]]) -> Resultado:
    if not codigo_aluno.strip():
        return Resultado(
            idx=q.idx,
            tipo=q.tipo,
            nota=0.0,
            status="erro",
            feedback="Resposta de código vazia.",
            detalhes=["Nenhum código foi fornecido pelo aluno."],
        )

    ok_sintaxe, erro_sintaxe = verificar_sintaxe_python(codigo_aluno)
    if not ok_sintaxe:
        return Resultado(
            idx=q.idx,
            tipo=q.tipo,
            nota=0.0,
            status="erro",
            feedback=f"Erro de sintaxe: {erro_sintaxe}",
            detalhes=["O código não compila em Python."],
        )

    if not testes:
        return Resultado(
            idx=q.idx,
            tipo=q.tipo,
            nota=0.0,
            status="erro",
            feedback="Não foi possível gerar nem localizar casos de teste confiáveis.",
            detalhes=["Sem testes para validar a solução."],
        )

    total = len(testes)
    passou = 0
    detalhes = []
    execucoes = []

    for i, teste in enumerate(testes, start=1):
        entrada = teste.get("entrada", "")
        saida_esperada = teste.get("saida", "")
        obs = teste.get("obs", "")

        execucao = executar_codigo_python(codigo_aluno, entrada, timeout=3)
        saida_obtida = normalizar_texto(execucao["stdout"])
        saida_esperada_norm = normalizar_texto(saida_esperada)

        sim = comparar_textos(saida_obtida, saida_esperada_norm)

        ok = False
        if execucao["timeout"]:
            ok = False
            motivo = "timeout"
        elif execucao["erro_execucao"]:
            ok = False
            motivo = execucao["erro_execucao"]
        elif sim >= LIMIAR_APROX:
            ok = True
            motivo = "ok"
        else:
            motivo = "saída diferente"

        if ok:
            passou += 1

        execucoes.append(
            {
                "teste": i,
                "entrada": entrada,
                "saida_esperada": saida_esperada_norm,
                "saida_obtida": saida_obtida,
                "obs": obs,
                "ok": ok,
                "motivo": motivo,
                "stderr": normalizar_texto(execucao["stderr"]),
                "returncode": execucao["returncode"],
                "timeout": execucao["timeout"],
            }
        )

        detalhes.append(
            f"Teste {i}: {'PASSOU' if ok else 'FALHOU'} "
            f"(similaridade={sim:.3f}, motivo={motivo})"
        )

    nota = (passou / total) * 10 if total else 0.0
    if passou == total:
        status = "ok"
        feedback = "Todos os testes passaram."
    elif passou >= max(1, total // 2):
        status = "parcial"
        feedback = f"{passou}/{total} testes passaram."
    else:
        status = "erro"
        feedback = f"Apenas {passou}/{total} testes passaram."

    return Resultado(
        idx=q.idx,
        tipo=q.tipo,
        nota=round(max(0.0, min(10.0, nota)), 2),
        status=status,
        feedback=feedback,
        detalhes=detalhes,
        testes_executados=execucoes,
    )


def avaliar_modificacao_com_llm(q: Questao, codigo_aluno: str) -> Resultado:
    """
    Avalia modificação combinando:
    - testes
    - checagem de aderência ao enunciado via LLM

    Importante:
    - Se o enunciado NÃO pedir saída na tela nem retorno,
      não penalize apenas por ausência de print/return.
    """
    testes = obter_testes(q)
    resultado_testes = avaliar_codigo_por_testes(q, codigo_aluno, testes)

    if not USAR_LLM:
        return resultado_testes

    precisa_saida = exige_saida_no_enunciado(q.enunciado)

    prompt = f"""
Você é um corretor rigoroso de questões de modificação de código.

Enunciado:
{q.enunciado}

O enunciado pede saída/retorno explícito?
{ "SIM" if precisa_saida else "NÃO" }

Código original de referência/contexto:
{q.codigo or "(não há)"}

Código enviado pelo aluno:
{codigo_aluno or "(vazio)"}

Sua tarefa:
- dizer se a modificação atende ao pedido do enunciado
- identificar requisitos que foram cumpridos e os que faltaram
- atribuir nota de 0 a 10
- use nota decimal quando fizer sentido
- se o enunciado NÃO pedir print/return, não penalize apenas por ausência de impressão

Retorne APENAS JSON válido com este formato:

{{
  "nota": 0,
  "status": "ok|parcial|erro",
  "cumpre_requisitos": true,
  "requisitos_atendidos": ["..."],
  "faltantes": ["..."],
  "feedback": "texto curto"
}}

Regras:
- seja objetivo
- seja coerente com o enunciado
- não avalie estilo, foque na exigência pedida
"""
    obj = chamar_llm_json(
        [
            {"role": "system", "content": "Você corrige modificações de código e devolve JSON válido."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.15,
        max_tokens=1200,
    )

    if isinstance(obj, dict):
        try:
            nota_llm = float(obj.get("nota", 0))
        except Exception:
            nota_llm = 0.0

        status_llm = str(obj.get("status", "parcial")).strip().lower()
        if status_llm not in {"ok", "parcial", "erro"}:
            status_llm = "parcial"

        finais = []
        at = obj.get("requisitos_atendidos", [])
        fl = obj.get("faltantes", [])

        if isinstance(at, list) and at:
            finais.append("Requisitos atendidos: " + "; ".join(str(x) for x in at[:6]))
        if isinstance(fl, list) and fl:
            finais.append("Faltantes: " + "; ".join(str(x) for x in fl[:6]))

        # Combinação ponderada: testes têm mais peso
        nota_final = (0.7 * resultado_testes.nota) + (0.3 * max(0.0, min(10.0, nota_llm)))

        if status_llm == "erro" and resultado_testes.status != "ok":
            status_final = "erro"
        elif resultado_testes.status == "ok" and status_llm == "ok":
            status_final = "ok"
        else:
            status_final = "parcial" if (resultado_testes.status != "ok" or status_llm != "ok") else "ok"

        feedback = str(obj.get("feedback", "")).strip() or resultado_testes.feedback

        return Resultado(
            idx=q.idx,
            tipo=q.tipo,
            nota=round(max(0.0, min(10.0, nota_final)), 2),
            status=status_final,
            feedback=feedback,
            detalhes=resultado_testes.detalhes + finais,
            testes_executados=resultado_testes.testes_executados,
        )

    return resultado_testes


# =========================
# Dispatcher
# =========================

def extrair_resposta_codigo(q: Questao) -> str:
    """
    Para questões de código, a resposta do aluno pode vir com markdown,
    texto e comentários. Aqui priorizamos blocos de código.
    """
    if not q.resposta_aluno:
        return ""

    codigo = extrair_codigo(q.resposta_aluno)
    return codigo.strip() or q.resposta_aluno.strip()


def corrigir_questao(q: Questao) -> Resultado:
    tipo = normalizar_tipo(q.tipo) or inferir_tipo(q.enunciado)

    if not tipo:
        tipo = inferir_tipo(q.enunciado)

    if tipo == "previsao":
        return avaliar_previsao(q)

    if tipo == "correcao":
        testes = obter_testes(q)
        codigo_aluno = extrair_resposta_codigo(q)
        return avaliar_codigo_por_testes(q, codigo_aluno, testes)

    if tipo == "modificacao":
        codigo_aluno = extrair_resposta_codigo(q)
        return avaliar_modificacao_com_llm(q, codigo_aluno)

    if tipo in {"justificativa", "descritiva"}:
        return avaliar_texto_llm(q)

    # fallback para qualquer tipo desconhecido
    if q.codigo or "```" in (q.enunciado or "") or "input(" in (q.enunciado or ""):
        codigo_aluno = extrair_resposta_codigo(q)
        testes = obter_testes(q)
        res = avaliar_codigo_por_testes(q, codigo_aluno, testes)
        if res.nota > 0:
            return res

    return avaliar_texto_llm(q)


# =========================
# Relatório
# =========================

def formatar_resultado(res: Resultado, q: Questao) -> str:
    linhas = []
    linhas.append(f"EXERCÍCIO {res.idx}")
    linhas.append(f"Tipo: {res.tipo}")
    linhas.append(f"Nota: {res.nota:.2f}/10")
    linhas.append(f"Status: {res.status}")
    linhas.append(f"Feedback: {res.feedback}")

    if q.enunciado:
        en = normalizar_texto(q.enunciado)
        if len(en) > 900:
            en = en[:900].rstrip() + "..."
        linhas.append("")
        linhas.append("Enunciado:")
        linhas.append(en)

    if res.saida_correta:
        linhas.append("")
        linhas.append("Saída calculada:")
        linhas.append(res.saida_correta if res.saida_correta else "(vazia)")

    if res.detalhes:
        linhas.append("")
        linhas.append("Detalhes:")
        for d in res.detalhes:
            linhas.append(f"- {d}")

    if res.testes_executados:
        linhas.append("")
        linhas.append("Testes executados:")
        for t in res.testes_executados:
            linhas.append(f"- Teste {t.get('teste', '?')}: {'PASSOU' if t.get('ok') else 'FALHOU'}")
            if t.get("obs"):
                linhas.append(f"  Obs: {t['obs']}")
            if t.get("entrada"):
                linhas.append("  Entrada:")
                linhas.append(textwrap.indent(normalizar_texto(str(t["entrada"])), "    "))
            if t.get("saida_esperada") is not None:
                linhas.append("  Saída esperada:")
                linhas.append(textwrap.indent(normalizar_texto(str(t["saida_esperada"])), "    "))
            if t.get("saida_obtida") is not None:
                linhas.append("  Saída obtida:")
                linhas.append(textwrap.indent(normalizar_texto(str(t["saida_obtida"])), "    "))
            if t.get("motivo"):
                linhas.append(f"  Motivo: {t['motivo']}")
            if t.get("timeout"):
                linhas.append("  Timeout: sim")
            if t.get("stderr"):
                linhas.append("  STDERR:")
                linhas.append(textwrap.indent(normalizar_texto(str(t["stderr"])), "    "))

    linhas.append("\n" + "=" * 72 + "\n")
    return "\n".join(linhas)


def gerar_relatorio(questoes: List[Questao], resultados: List[Resultado]) -> str:
    media = sum(r.nota for r in resultados) / max(1, len(resultados))
    aprovadas = sum(1 for r in resultados if r.nota >= 7.0)

    linhas = []
    linhas.append("CORREÇÃO AUTOMÁTICA")
    linhas.append(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    linhas.append(f"Total de questões: {len(resultados)}")
    linhas.append(f"Média geral: {media:.2f}/10")
    linhas.append(f"Questões com nota >= 7.0: {aprovadas}")
    linhas.append("=" * 72)
    linhas.append("")

    for q, r in zip(questoes, resultados):
        linhas.append(formatar_resultado(r, q))

    return "\n".join(linhas).rstrip() + "\n"


# =========================
# Main
# =========================

def main() -> int:
    try:
        questoes = carregar_questoes(ARQUIVO_ENTRADA)
    except Exception as e:
        print(f"Erro ao ler a entrada: {e}", file=sys.stderr)
        return 1

    if not questoes:
        print("Nenhuma questão encontrada no arquivo de entrada.", file=sys.stderr)
        return 1

    resultados: List[Resultado] = []
    for q in questoes:
        try:
            resultados.append(corrigir_questao(q))
        except Exception as e:
            resultados.append(
                Resultado(
                    idx=q.idx,
                    tipo=q.tipo or "desconhecido",
                    nota=0.0,
                    status="erro",
                    feedback=f"Erro interno ao corrigir a questão: {e}",
                    detalhes=[traceback.format_exc(limit=3)],
                )
            )

    relatorio = gerar_relatorio(questoes, resultados)

    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    ARQUIVO_SAIDA.write_text(relatorio, encoding="utf-8")

    print(f"Correção salva em: {ARQUIVO_SAIDA}")
    print(f"Média geral: {sum(r.nota for r in resultados) / len(resultados):.2f}/10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())