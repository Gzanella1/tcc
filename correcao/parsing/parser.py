#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
parsing/parser.py

Leitura e parsing do arquivo de questões.
Suporta formato JSON e formato de texto em blocos separados por delimitadores.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from models.questao import Questao
from utils.text import extrair_codigo, normalizar_label, normalizar_texto, sem_acentos
from utils.tipo import normalizar_tipo, inferir_tipo


# ─── Mapeamento de aliases de campos ─────────────────────────────────────────
ALIASES_CAMPOS: Dict[str, str] = {
    "id":                   "id",
    "idx":                  "id",
    "tipo":                 "tipo",
    "enunciado":            "enunciado",
    "questao":              "enunciado",
    "questão":              "enunciado",
    "pergunta":             "enunciado",
    "texto":                "enunciado",
    "resposta do aluno":    "resposta_aluno",
    "resposta aluno":       "resposta_aluno",
    "resposta":             "resposta_aluno",
    "answer":               "resposta_aluno",
    "solucao do aluno":     "resposta_aluno",
    "solução do aluno":     "resposta_aluno",
    "resposta esperada":    "resposta_referencia",
    "resposta modelo":      "resposta_referencia",
    "modelo":               "resposta_referencia",
    "gabarito":             "resposta_referencia",
    "rubrica":              "rubrica",
    "criterios":            "rubrica",
    "critérios":            "rubrica",
    "codigo":               "codigo_base",
    "código":               "codigo_base",
    "programa":             "codigo_base",
    "trecho de codigo":     "codigo_base",
    "trecho de código":     "codigo_base",
    "codigo base":          "codigo_base",
    "código base":          "codigo_base",
    "codigo original":      "codigo_base",
    "código original":      "codigo_base",
    "codigo da questao":    "codigo_base",
    "código da questão":    "codigo_base",
    "codigo aluno":         "codigo_aluno",
    "código aluno":         "codigo_aluno",
    "codigo do aluno":      "codigo_aluno",
    "código do aluno":      "codigo_aluno",
    "resposta em codigo":   "codigo_aluno",
    "resposta em código":   "codigo_aluno",
    "stdin":                "entrada",
    "input":                "entrada",
    "entrada":              "entrada",
    "saida":                "saida_esperada",
    "saída":                "saida_esperada",
    "output":               "saida_esperada",
    "saida esperada":       "saida_esperada",
    "saída esperada":       "saida_esperada",
    "testes":               "testes",
    "casos de teste":       "testes",
    "formato da resposta":  "resposta_formato",
    "resposta formato":     "resposta_formato",
    "resposta_formato":     "resposta_formato",
    # ── Fase 3.1: rastreabilidade da origem da pergunta ─────────────────────
    # Campos de ORIGEM nunca viram codigo_base; são preservados em extras.
    "codigo aluno origem":       "codigo_aluno_origem",
    "enunciado origem":          "enunciado_origem",
    "enunciado original":        "enunciado_origem",
}

_ALIASES_NORMALIZADOS = {
    normalizar_label(chave): valor for chave, valor in ALIASES_CAMPOS.items()
}

# Linha separadora de código (ex: ----------------------------------------)
_LINHA_SEPARADORA = re.compile(r"^-{4,}\s*$")

# ── Fase 3.1: rótulos do novo formato de geração ────────────────────────────
# Identificam a origem da pergunta. São reconhecidos apenas ANTES do
# cabeçalho "N - [TIPO]", e nunca convertem o conteúdo em codigo_base.
_ROTULO_ENUNCIADO_ORIGEM = re.compile(
    r"(?i)^enunciado\s+original\s*:\s*$"
)
_ROTULO_CODIGO_ALUNO_ORIGEM = re.compile(
    r"(?i)^(?:o\s+)?c[oó]digo\s+do\s+aluno\s+que\s+originou\s+esta\s+pergunta\s*:?\s*$"
)


def _campo(item: Dict[str, Any], *nomes: str, default: Any = "") -> Any:
    """Busca o primeiro campo existente em um dicionario ja normalizado."""
    for nome in nomes:
        if nome in item and item[nome] is not None:
            return item[nome]
    return default


def _normalizar_item_json(item: Dict[str, Any]) -> Dict[str, Any]:
    """Converte aliases de campos JSON para o contrato canonico de Questao."""
    normalizado: Dict[str, Any] = {}
    extras: Dict[str, Any] = {}

    for chave, valor in item.items():
        canonica = _ALIASES_NORMALIZADOS.get(normalizar_label(str(chave)))
        if canonica:
            normalizado[canonica] = valor
        else:
            extras[chave] = valor

    normalizado["_extras"] = extras
    return normalizado


def _rubrica_padrao(tipo: str) -> str:
    """Rubrica minima para preservar o contrato em entradas textuais legadas."""
    if tipo == "descritiva":
        return (
            "Avaliar se a resposta descreve corretamente o fluxo do codigo, "
            "as condicoes avaliadas e os resultados produzidos."
        )
    if tipo == "justificativa":
        return (
            "Avaliar se a resposta justifica a decisao de implementacao com "
            "base no comportamento do codigo."
        )
    if tipo == "correcao":
        return (
            "Avaliar se a resposta identifica corretamente o problema pedido "
            "no enunciado e explica seu impacto."
        )
    return ""


def _enunciado_pede_texto(tipo: str, enunciado: str) -> bool:
    """Detecta perguntas em que a resposta esperada e textual."""
    if tipo in {"descritiva", "justificativa"}:
        return True

    if tipo != "correcao":
        return False

    e = sem_acentos((enunciado or "").lower())
    return any(p in e for p in [
        "qual e o erro",
        "qual e o problema",
        "como corrigir",
        "explique o erro",
        "o que esta errado",
        "por que",
        "justifique",
    ])


def _resposta_tem_codigo_explicito(resposta: str) -> bool:
    """Verifica se a resposta contem cercas markdown ou rotulo de codigo."""
    texto = resposta or ""
    return bool(
        "```" in texto
        or re.search(r"(?im)^\s*(?:seu\s+)?c[oó]digo\s*:?\s*$", texto)
    )


def _extrair_codigo_resposta(resposta: str, tipo: str, enunciado: str) -> str:
    """
    Extrai codigo entregue pelo aluno sem confundir explicacoes textuais
    que apenas mencionam pequenos trechos de codigo.
    """
    texto = normalizar_texto(resposta)
    if not texto:
        return ""

    if "```" in texto:
        return normalizar_texto(extrair_codigo(texto))

    m = re.search(
        r"(?im)^\s*(?:seu\s+)?c[oó]digo\s*:?\s*$",
        texto,
    )
    if m:
        return normalizar_texto(texto[m.end():])

    return ""


def _formato_resposta(tipo: str, enunciado: str, resposta: str, codigo_aluno: str) -> str:
    if codigo_aluno:
        return "codigo"
    if _enunciado_pede_texto(tipo, enunciado):
        return "texto"
    if _resposta_tem_codigo_explicito(resposta):
        return "codigo"
    return "texto" if resposta else ""


def carregar_arquivo_texto(path: Path) -> str:
    """Lê o arquivo de entrada e retorna seu conteúdo como string."""
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def split_exercicios(texto: str) -> List[str]:
    """
    Divide o texto em blocos de exercícios.

    Suporta dois formatos:
        Formato 1 (linha única):
            ========== Exercicio gerado ... ==========

        Formato 2 (três linhas, com acento):
            ============================================================
            Exercício gerado com base na sua resposta da questão X
            ============================================================
    """
    # Formato 1: tudo numa linha
    padrao_linha = re.compile(r"(?i)=+\s*exerc[ií]cio.*?=+")
    if padrao_linha.search(texto):
        partes = re.split(padrao_linha, texto)
        return [p.strip() for p in partes if p.strip()]

    # Formato 2: cabeçalho de três linhas (===\ntexto\n===)
    padrao_bloco = re.compile(r"={3,}\n[^\n]*exerc[ií]cio[^\n]*\n={3,}", re.IGNORECASE)
    if padrao_bloco.search(texto):
        partes = re.split(padrao_bloco, texto)
        return [p.strip() for p in partes if p.strip()]

    # Sem separador reconhecido → trata como bloco único
    return [texto.strip()]


def parse_tests_field(texto: str) -> List[Dict[str, str]]:
    """
    Parseia o campo de testes, suportando:
    - Lista JSON: [{"entrada": ..., "saida": ...}, ...]
    - Linhas no formato: entrada => saida
    - Linhas no formato: entrada | saida
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
                        testes.append({
                            "entrada": str(item.get("entrada", item.get("input", ""))),
                            "saida":   str(item.get("saida",   item.get("output", ""))),
                            "obs":     str(item.get("obs",     item.get("descricao", ""))),
                        })
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
            testes.append({"entrada": esquerda.strip(), "saida": direita.strip(), "obs": ""})
        elif "|" in linha:
            esquerda, direita = linha.split("|", 1)
            testes.append({"entrada": esquerda.strip(), "saida": direita.strip(), "obs": ""})

    return testes


def parse_block(block: str, idx: int) -> Questao:
    """
    Parseia um bloco de texto no formato gerado pelo sistema:

        N - [TIPO] Enunciado da questão

        Seu código:
        ----------------------------------------
        <código python>
        ----------------------------------------

        resposta N -
        <resposta do aluno>

    Regras de captura:
    - Cabeçalho da questão: "N - [TIPO] texto"
    - Início do código: linha que contenha "código:" ou "seu código:" (case-insensitive)
    - Delimitadores de código: linhas com 4+ hífens (ignoradas, não entram no código)
    - Início da resposta: "resposta N -" ou "resposta N:"
    """
    linhas = block.splitlines()

    tipo                = ""
    enunciado_linhas    = []
    resposta            = []
    codigo              = []
    capturando_resposta = False
    capturando_codigo   = False
    dentro_bloco_codigo = False   # True entre os dois separadores ---

    # ── Fase 3.1: origem da pergunta (formato novo de geração) ───────────
    enunciado_origem_linhas = []
    codigo_origem_linhas    = []
    capturando_enunciado_origem = False
    capturando_codigo_origem    = False
    dentro_bloco_codigo_origem  = False

    for linha in linhas:
        linha_strip = linha.strip()

        # ── Cabeçalho: "N - [TIPO] enunciado" ────────────────────────────────
        m = re.match(r"\d+\s*-\s*\[(\w+)\]\s*(.+)", linha_strip, re.IGNORECASE)
        if m:
            tipo      = m.group(1)
            enunciado_linhas = [m.group(2)]
            capturando_resposta = False
            capturando_codigo   = False
            dentro_bloco_codigo = False
            capturando_enunciado_origem = False
            capturando_codigo_origem    = False
            dentro_bloco_codigo_origem  = False
            continue

        # ── Rótulos da origem (aparecem antes do cabeçalho no formato novo) ──
        if not tipo:
            if _ROTULO_ENUNCIADO_ORIGEM.match(linha_strip):
                capturando_enunciado_origem = True
                continue

            if capturando_enunciado_origem:
                if linha_strip:
                    enunciado_origem_linhas.append(linha_strip)
                else:
                    capturando_enunciado_origem = False
                continue

            if _ROTULO_CODIGO_ALUNO_ORIGEM.match(linha_strip):
                capturando_codigo_origem    = True
                dentro_bloco_codigo_origem  = False
                continue

            if capturando_codigo_origem:
                if _LINHA_SEPARADORA.match(linha_strip):
                    if not dentro_bloco_codigo_origem:
                        dentro_bloco_codigo_origem = True    # primeiro --- → abre
                    else:
                        dentro_bloco_codigo_origem = False   # segundo --- → fecha
                        capturando_codigo_origem   = False
                    continue
                if dentro_bloco_codigo_origem:
                    codigo_origem_linhas.append(linha)
                    continue
                continue

        # ── Continuação do enunciado (linhas antes do "Seu código:" ou resposta) ──
        if tipo and not capturando_codigo and not capturando_resposta:
            # Detecta início do bloco de código
            if re.match(r"(?i)^(seu\s+)?c[oó]digo\s*:\s*$", linha_strip):
                capturando_codigo   = True
                dentro_bloco_codigo = False
                continue
            # Detecta início da resposta
            m2 = re.match(r"resposta\s*\d+\s*[-:]\s*(.*)", linha_strip, re.IGNORECASE)
            if m2:
                capturando_resposta = True
                capturando_codigo   = False
                if m2.group(1).strip():
                    resposta.append(m2.group(1).strip())
                continue
            # Linha de enunciado adicional (ex: "Considere também o caso em que...")
            if linha_strip:
                enunciado_linhas.append(linha_strip)
            continue

        # ── Captura de código ─────────────────────────────────────────────────
        if capturando_codigo:
            # Separador --- abre ou fecha o bloco de código
            if _LINHA_SEPARADORA.match(linha_strip):
                if not dentro_bloco_codigo:
                    dentro_bloco_codigo = True   # primeiro --- → abre
                else:
                    dentro_bloco_codigo = False  # segundo --- → fecha
                    capturando_codigo   = False
                continue

            # Dentro do bloco delimitado por ---
            if dentro_bloco_codigo:
                codigo.append(linha)
                continue

            # Sem delimitadores: captura direto até encontrar resposta
            m2 = re.match(r"resposta\s*\d+\s*[-:]\s*(.*)", linha_strip, re.IGNORECASE)
            if m2:
                capturando_codigo   = False
                capturando_resposta = True
                if m2.group(1).strip():
                    resposta.append(m2.group(1).strip())
                continue

            codigo.append(linha)
            continue

        # ── Captura de resposta ───────────────────────────────────────────────
        if capturando_resposta:
            if re.match(r"(?i)^(seu\s+)?c[oó]digo\s*:\s*$", linha_strip):
                capturando_resposta = False
                capturando_codigo = True
                continue
            m2 = re.match(r"resposta\s*\d+\s*[-:]\s*(.*)", linha_strip, re.IGNORECASE)
            if m2:
                # Nova questão dentro do mesmo bloco (não deve acontecer, mas protege)
                if m2.group(1).strip():
                    resposta.append(m2.group(1).strip())
                continue
            resposta.append(linha)

    enunciado_bruto = "\n".join(enunciado_linhas)
    enunciado = normalizar_texto(enunciado_bruto)
    tipo_norm = normalizar_tipo(tipo)
    resposta_norm = normalizar_texto("\n".join(resposta))
    codigo_base = normalizar_texto("\n".join(codigo))
    codigo_aluno = _extrair_codigo_resposta(resposta_norm, tipo_norm, enunciado)
    resposta_formato = _formato_resposta(tipo_norm, enunciado, resposta_norm, codigo_aluno)

    # ── Fase 3.1: origem preservada em extras, NUNCA em codigo_base ─────────
    extras: Dict[str, Any] = {}
    if enunciado_origem_linhas:
        extras["enunciado_origem"] = normalizar_texto("\n".join(enunciado_origem_linhas))
    if codigo_origem_linhas:
        extras["codigo_aluno_origem"] = normalizar_texto("\n".join(codigo_origem_linhas))
    if resposta_formato:
        extras["resposta_formato"] = resposta_formato

    return Questao(
        idx=idx,
        tipo=tipo_norm,
        enunciado=enunciado,
        resposta_aluno=resposta_norm,
        resposta_referencia="",
        rubrica=_rubrica_padrao(tipo_norm),
        codigo_base=codigo_base,
        codigo_aluno=codigo_aluno,
        saida_esperada="",
        codigo=codigo_base,
        entrada=extrair_entradas(enunciado_bruto, tipo_norm),
        saida="",
        testes=[],
        extras=extras,
    )


def carregar_questoes(path: Path) -> List[Questao]:
    """
    Carrega questões a partir de um arquivo.
    Tenta JSON primeiro; se falhar, usa o parser de blocos de texto.
    """
    texto = carregar_arquivo_texto(path).strip()
    if not texto:
        return []

    # ── Tentativa JSON ────────────────────────────────────────────────────────
    if texto[0] in "{[":
        try:
            obj = json.loads(texto)
            if isinstance(obj, dict):
                itens = (
                    obj.get("perguntas")
                    or obj.get("questoes")
                    or obj.get("questions")
                    or []
                )
            elif isinstance(obj, list):
                itens = obj
            else:
                itens = []

            questoes = []
            for i, item in enumerate(itens, start=1):
                if not isinstance(item, dict):
                    continue

                dados = _normalizar_item_json(item)
                testes_raw = dados.get("testes")
                testes_str = (
                    json.dumps(testes_raw, ensure_ascii=False)
                    if testes_raw is not None
                    else ""
                )
                enunciado = normalizar_texto(str(_campo(dados, "enunciado")))
                tipo_norm = (
                    normalizar_tipo(str(_campo(dados, "tipo")))
                    or inferir_tipo(enunciado)
                )
                resposta_aluno = normalizar_texto(str(_campo(dados, "resposta_aluno")))
                codigo_base = normalizar_texto(str(_campo(dados, "codigo_base")))
                codigo_aluno = normalizar_texto(str(_campo(dados, "codigo_aluno")))

                if not codigo_aluno:
                    codigo_aluno = _extrair_codigo_resposta(
                        resposta_aluno,
                        tipo_norm,
                        enunciado,
                    )

                resposta_formato = str(
                    _campo(dados, "resposta_formato", default="")
                ).strip().lower()
                if not resposta_formato:
                    resposta_formato = _formato_resposta(
                        tipo_norm,
                        enunciado,
                        resposta_aluno,
                        codigo_aluno,
                    )

                # ── Fase 3.1: origem da pergunta (rastreabilidade) ───────────
                # Campos canônicos já normalizados pelos aliases; nunca
                # alimentam codigo_base.
                enunciado_origem = normalizar_texto(
                    str(_campo(dados, "enunciado_origem"))
                )
                codigo_aluno_origem = normalizar_texto(
                    str(_campo(dados, "codigo_aluno_origem"))
                )
                tem_origem = bool(enunciado_origem or codigo_aluno_origem)

                extras_origem = {}
                if enunciado_origem:
                    extras_origem["enunciado_origem"] = enunciado_origem
                if codigo_aluno_origem:
                    extras_origem["codigo_aluno_origem"] = codigo_aluno_origem

                q = Questao(
                    idx=int(_campo(dados, "id", default=i)),
                    tipo=tipo_norm,
                    enunciado=enunciado,
                    resposta_aluno=resposta_aluno,
                    resposta_referencia=normalizar_texto(str(_campo(dados, "resposta_referencia"))),
                    rubrica=normalizar_texto(str(_campo(dados, "rubrica"))) or _rubrica_padrao(tipo_norm),
                    codigo_base=codigo_base,
                    codigo_aluno=codigo_aluno,
                    saida_esperada=normalizar_texto(str(_campo(dados, "saida_esperada"))),
                    codigo=codigo_base,
                    entrada=normalizar_texto(str(_campo(dados, "entrada"))) or extrair_entradas(enunciado, tipo_norm),
                    saida=normalizar_texto(str(_campo(dados, "saida_esperada"))),
                    testes=parse_tests_field(testes_str),
                    extras={
                        **dados.get("_extras", {}),
                        **extras_origem,
                        **({"resposta_formato": resposta_formato} if resposta_formato else {}),
                    },
                )

                # Fallback legado: extrai código do enunciado/resposta APENAS
                # quando não há origem declarada — senão o código que originou
                # a pergunta seria convertido em codigo_base indevidamente.
                if not tem_origem and not q.codigo_base and not q.codigo:
                    codigo_extraido = extrair_codigo(q.enunciado or q.resposta_aluno)
                    if codigo_extraido != (q.enunciado or q.resposta_aluno):
                        q.codigo_base = normalizar_texto(codigo_extraido)
                        q.codigo = q.codigo_base

                questoes.append(q)

            if questoes:
                return questoes
        except Exception:
            pass

    # ── Fallback: blocos de texto ─────────────────────────────────────────────
    blocos = split_exercicios(texto)
    if len(blocos) == 1:
        return [parse_block(blocos[0], 1)]

    return [parse_block(bloco, i + 1) for i, bloco in enumerate(blocos)]

def extrair_entradas(enunciado: str, tipo: str = "") -> str:
    """
    Extrai entrada estruturada de enunciados legados.

    Suporta:
    - linhas "Entrada: ..."
    - questoes de previsao com vetor/lista no enunciado, como
      "Dado o vetor [2, 3, 4]".
    """
    entradas = re.findall(r"(?im)^\s*Entrada\s*:\s*(.+?)\s*$", enunciado)
    if entradas:
        return "\n".join(e.strip() for e in entradas if e.strip()) + "\n"

    texto = enunciado or ""
    texto_norm = sem_acentos(texto.lower())
    if tipo == "previsao" and "vetor" in texto_norm:
        m = re.search(r"\[([^\]]+)\]", texto)
        if m:
            valores = [
                v.strip().strip("\"'")
                for v in m.group(1).split(",")
                if v.strip()
            ]
            if valores:
                return "\n".join(valores) + "\n"

    return ""
