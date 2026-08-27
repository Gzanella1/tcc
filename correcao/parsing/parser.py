#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
parsing/parser.py

Leitura e parsing do arquivo de questões.
Suporta formato JSON e formato de texto em blocos separados por delimitadores.

Contrato canônico produzido:
- codigo_base           : código ANTERIOR do aluno (contexto/apoio; nunca
                          gabarito). Rótulos reconhecidos: "Seu código:",
                          "Código:", "Código-base:", "Código original:",
                          além dos aliases JSON correspondentes e dos rótulos
                          de origem da geração ("Código do aluno que originou
                          esta pergunta:").
- codigo_aluno_resposta : o NOVO código entregue pelo aluno como resposta.
                          Extraído da região de resposta (blocos delimitados
                          por "---", cercas ``` ou rótulo "Código:").
                          Nunca alimenta codigo_base.
- entradaTestes         : entradas declaradas/enumeradas no enunciado.
- extras                : apenas metadados/rastreabilidade sem campo canônico
                          (ex.: enunciado_origem). Nenhum dado com campo
                          canônico é espelhado aqui.
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
    # Código ANTERIOR do aluno (contexto) — todos os sinônimos convergem para
    # o campo canônico codigo_base, incluindo aliases legados de origem.
    "codigo":                    "codigo_base",
    "código":                    "codigo_base",
    "programa":                  "codigo_base",
    "trecho de codigo":          "codigo_base",
    "trecho de código":          "codigo_base",
    "codigo base":               "codigo_base",
    "código base":               "codigo_base",
    "codigo-base":               "codigo_base",
    "código-base":               "codigo_base",
    "codigo original":           "codigo_base",
    "código original":           "codigo_base",
    "codigo da questao":         "codigo_base",
    "código da questão":         "codigo_base",
    "codigo anterior do aluno":  "codigo_base",
    "código anterior do aluno":  "codigo_base",
    "codigo aluno origem":       "codigo_base",
    "NOVO":                      None,  # marcador substituído abaixo
    "codigo aluno":              "codigo_aluno_resposta",
    "código aluno":              "codigo_aluno_resposta",
    "codigo do aluno":           "codigo_aluno_resposta",
    "código do aluno":           "codigo_aluno_resposta",
    "resposta em codigo":        "codigo_aluno_resposta",
    "resposta em código":        "codigo_aluno_resposta",
    "stdin":                     "entradaTestes",
    "input":                     "entradaTestes",
    "entrada":                   "entradaTestes",
    "saida":                     "saidaTestes",
    "saída":                     "saidaTestes",
    "output":                    "saida_esperada",
    "saida esperada":            "saida_esperada",
    "saída esperada":            "saida_esperada",
    "testes":                    "testes",
    "casos de teste":            "testes",
    "formato da resposta":       "resposta_formato",
    "resposta formato":          "resposta_formato",
    "resposta_formato":          "resposta_formato",
    # ── Rastreabilidade da origem da pergunta ────────────────────────────────
    # enunciado_origem não possui campo canônico próprio (é proveniência do
    # enunciado que gerou a pergunta) e permanece em extras.
    "enunciado origem":          "enunciado_origem",
    "enunciado original":        "enunciado_origem",
}
ALIASES_CAMPOS.pop("NOVO", None)

_ALIASES_NORMALIZADOS = {
    normalizar_label(chave): valor for chave, valor in ALIASES_CAMPOS.items()
}

# Linha separadora de código (ex: ----------------------------------------)
_LINHA_SEPARADORA = re.compile(r"^-{4,}\s*$")

# Rótulos que abrem um bloco de CÓDIGO-BASE após o cabeçalho:
# "Seu código:", "Código:", "Código-base:", "código base:", "Código original:"
_RE_ROTULO_CODIGO_BASE = re.compile(
    r"(?i)^(?:seu\s+)?c[oó]digo(?:[\s-]+base|[\s-]+original)?\s*:\s*$"
)

# Mesmo padrão, usado DENTRO da região de resposta: ali ele identifica o
# código ENTREGUE pelo aluno, nunca o código-base.
_RE_ROTULO_CODIGO_RESPOSTA = re.compile(
    r"(?i)^(?:seu\s+)?c[oó]digo\s*:\s*$"
)

# ── Rótulos de codigo_base exportados pela geração (antes do cabeçalho) ───────
_ROTULO_ENUNCIADO_ORIGEM = re.compile(
    r"(?i)^enunciado\s+original\s*:\s*$"
)
_ROTULO_CODIGO_BASE_ORIGEM = re.compile(
    r"(?i)^(?:o\s+)?(?:c[oó]digo\s+do\s+aluno\s+que\s+originou\s+esta\s+pergunta|c[oó]digo[- ]?base)\s*:?\s*$"
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
        or _RE_ROTULO_CODIGO_RESPOSTA.match(texto.strip())
    )


def _extrair_codigo_resposta(resposta: str) -> str:
    """
    Extrai codigo entregue pelo aluno a partir de uma resposta bruta,
    sem confundir explicacoes textuais que apenas mencionam trechos.
    """
    texto = normalizar_texto(resposta)
    if not texto:
        return ""

    if "```" in texto:
        return normalizar_texto(extrair_codigo(texto))

    m = _RE_ROTULO_CODIGO_RESPOSTA.search(texto)
    if m:
        return normalizar_texto(texto[m.end():])

    return ""


def _formato_resposta(tipo: str, enunciado: str, resposta: str, codigo_aluno_resposta: str) -> str:
    if codigo_aluno_resposta:
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

        [Enunciado original:
        <enunciado do exercicio de origem>]

        [Código-base:
        ----------------------------------------
        <código anterior do aluno>
        ----------------------------------------]

        N - [TIPO] Enunciado da questão

        Seu código: | Código-base:
        ----------------------------------------
        <código python>
        ----------------------------------------

        resposta N -
        <resposta do aluno>

    Regras de captura (contrato canônico):
    - Cabeçalho da questão: "N - [TIPO] texto".
    - Após o cabeçalho, rótulos "Seu código:", "Código:", "Código-base:",
      "Código original:" abrem captura de CODIGO-BASE (contexto).
    - Dentro da região de resposta, rótulos "Código:" ou blocos delimitados
      por "---" capturam o CODIGO DA RESPOSTA (nunca viram codigo_base).
    - Delimitadores: linhas com 4+ hífens.
    - Início da resposta: "resposta N -" ou "resposta N:".
    """
    linhas = block.splitlines()

    tipo             = ""
    enunciado_linhas = []
    resposta_linhas       = []   # parte TEXTUAL da resposta
    resposta_cod_linhas   = []   # código entregue pelo aluno na resposta
    codigo_linhas         = []   # código-base (contexto)

    # Estados da máquina de captura
    capturando_enunciado = False
    capturando_codigo    = False   # capturando codigo_base
    capturando_resposta  = False
    modo_resposta_codigo = False   # dentro da resposta: coletando código?
    codigo_via_rotulo    = False   # código da resposta aberto por rótulo
    dentro_bloco_codigo  = False   # True entre dois separadores ---

    # ── Origem da pergunta (antes do cabeçalho) ──────────────────────────────
    enunciado_origem_linhas = []
    codigo_origem_linhas    = []
    capturando_enunciado_origem = False
    capturando_codigo_origem    = False
    dentro_bloco_codigo_origem  = False

    def _fechar_codigo_resposta() -> None:
        nonlocal modo_resposta_codigo, codigo_via_rotulo, dentro_bloco_codigo
        modo_resposta_codigo = False
        codigo_via_rotulo = False
        dentro_bloco_codigo = False

    for linha in linhas:
        linha_strip = linha.strip()

        # ── Cabeçalho: "N - [TIPO] enunciado" ────────────────────────────────
        m = re.match(r"\d+\s*-\s*\[(\w+)\]\s*(.+)", linha_strip, re.IGNORECASE)
        if m:
            tipo              = m.group(1)
            enunciado_linhas  = [m.group(2)]
            capturando_enunciado = True
            capturando_codigo    = False
            capturando_resposta  = False
            _fechar_codigo_resposta()
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

            if _ROTULO_CODIGO_BASE_ORIGEM.match(linha_strip):
                capturando_codigo_origem   = True
                dentro_bloco_codigo_origem = False
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

        # ── Continuação do enunciado (antes do código ou da resposta) ────────
        if tipo and not capturando_codigo and not capturando_resposta:
            if _RE_ROTULO_CODIGO_BASE.match(linha_strip):
                capturando_enunciado = False
                capturando_codigo    = True
                dentro_bloco_codigo  = False
                continue
            m2 = re.match(r"resposta\s*\d+\s*[-:]\s*(.*)", linha_strip, re.IGNORECASE)
            if m2:
                capturando_enunciado = False
                capturando_resposta  = True
                _fechar_codigo_resposta()
                if m2.group(1).strip():
                    resposta_linhas.append(m2.group(1).strip())
                continue
            if linha_strip:
                enunciado_linhas.append(linha_strip)
            continue

        # ── Captura de código-base ────────────────────────────────────────────
        if capturando_codigo:
            if _LINHA_SEPARADORA.match(linha_strip):
                if not dentro_bloco_codigo:
                    dentro_bloco_codigo = True   # primeiro --- → abre
                else:
                    dentro_bloco_codigo = False  # segundo --- → fecha
                    capturando_codigo   = False
                continue

            if dentro_bloco_codigo:
                codigo_linhas.append(linha)
                continue

            m2 = re.match(r"resposta\s*\d+\s*[-:]\s*(.*)", linha_strip, re.IGNORECASE)
            if m2:
                capturando_codigo   = False
                capturando_resposta = True
                _fechar_codigo_resposta()
                if m2.group(1).strip():
                    resposta_linhas.append(m2.group(1).strip())
                continue

            codigo_linhas.append(linha)
            continue

        # ── Captura de resposta ───────────────────────────────────────────────
        if capturando_resposta:
            m2 = re.match(r"resposta\s*\d+\s*[-:]\s*(.*)", linha_strip, re.IGNORECASE)
            if m2:
                # Nova questão dentro do mesmo bloco (proteção)
                if m2.group(1).strip():
                    resposta_linhas.append(m2.group(1).strip())
                continue

            # Rótulo "Código:" dentro da resposta → código ENTREGUE pelo aluno
            if _RE_ROTULO_CODIGO_RESPOSTA.match(linha_strip):
                _fechar_codigo_resposta()
                modo_resposta_codigo = True
                codigo_via_rotulo    = True
                dentro_bloco_codigo  = False
                continue

            # Separador dentro da resposta delimita bloco de código entregue
            if _LINHA_SEPARADORA.match(linha_strip):
                if not modo_resposta_codigo:
                    # Abre bloco implícito de código da resposta
                    modo_resposta_codigo = True
                    codigo_via_rotulo    = False
                    dentro_bloco_codigo  = True
                elif codigo_via_rotulo and not dentro_bloco_codigo:
                    # Primeiro --- depois de um rótulo: abre o bloco
                    dentro_bloco_codigo = True
                elif codigo_via_rotulo and dentro_bloco_codigo:
                    _fechar_codigo_resposta()
                else:
                    # Fecha bloco implícito aberto por ---
                    _fechar_codigo_resposta()
                continue

            if modo_resposta_codigo:
                resposta_cod_linhas.append(linha)
            else:
                resposta_linhas.append(linha)

    enunciado_bruto = "\n".join(enunciado_linhas)
    enunciado = normalizar_texto(enunciado_bruto)
    tipo_norm = normalizar_tipo(tipo)

    resposta_texto_norm = normalizar_texto("\n".join(resposta_linhas))
    codigo_da_resposta = normalizar_texto("\n".join(resposta_cod_linhas))
    if not codigo_da_resposta:
        # Último recurso legado: cercas ``` ou rótulo dentro do texto bruto.
        codigo_da_resposta = _extrair_codigo_resposta(resposta_texto_norm)

    codigo_base = normalizar_texto("\n".join(codigo_linhas))

    # Origem pré-cabeçalho: o código anterior do aluno É o codigo_base.
    if not codigo_base and codigo_origem_linhas:
        codigo_base = normalizar_texto("\n".join(codigo_origem_linhas))

    resposta_formato = _formato_resposta(tipo_norm, enunciado, resposta_texto_norm, codigo_da_resposta)

    extras: Dict[str, Any] = {}
    if enunciado_origem_linhas:
        extras["enunciado_origem"] = normalizar_texto("\n".join(enunciado_origem_linhas))
    if resposta_formato:
        extras["resposta_formato"] = resposta_formato

    return Questao(
        idx=idx,
        tipo=tipo_norm,
        enunciado=enunciado,
        resposta_aluno=resposta_texto_norm,
        resposta_referencia="",
        rubrica=_rubrica_padrao(tipo_norm),
        codigo_base=codigo_base,
        codigo_aluno_resposta=codigo_da_resposta,
        saida_esperada="",
        entradaTestes=extrair_entradas(enunciado_bruto, tipo_norm),
        saidaTestes="",
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
                codigo_aluno_resposta = normalizar_texto(
                    str(_campo(dados, "codigo_aluno_resposta"))
                )

                if not codigo_aluno_resposta:
                    codigo_aluno_resposta = _extrair_codigo_resposta(resposta_aluno)

                resposta_formato = str(
                    _campo(dados, "resposta_formato", default="")
                ).strip().lower()
                if not resposta_formato:
                    resposta_formato = _formato_resposta(
                        tipo_norm,
                        enunciado,
                        resposta_aluno,
                        codigo_aluno_resposta,
                    )

                # Proveniência do enunciado (sem campo canônico → extras).
                enunciado_origem = normalizar_texto(
                    str(_campo(dados, "enunciado_origem"))
                )

                extras: Dict[str, Any] = dict(dados.get("_extras", {}))
                if enunciado_origem:
                    extras["enunciado_origem"] = enunciado_origem
                if resposta_formato:
                    extras["resposta_formato"] = resposta_formato

                q = Questao(
                    idx=int(_campo(dados, "id", default=i)),
                    tipo=tipo_norm,
                    enunciado=enunciado,
                    resposta_aluno=resposta_aluno,
                    resposta_referencia=normalizar_texto(str(_campo(dados, "resposta_referencia"))),
                    rubrica=normalizar_texto(str(_campo(dados, "rubrica"))) or _rubrica_padrao(tipo_norm),
                    codigo_base=codigo_base,
                    codigo_aluno_resposta=codigo_aluno_resposta,
                    saida_esperada=normalizar_texto(str(_campo(dados, "saida_esperada"))),
                    entradaTestes=(
                        normalizar_texto(str(_campo(dados, "entradaTestes")))
                        or extrair_entradas(enunciado, tipo_norm)
                    ),
                    saidaTestes=normalizar_texto(str(_campo(dados, "saidaTestes"))),
                    testes=parse_tests_field(testes_str),
                    extras=extras,
                )
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
