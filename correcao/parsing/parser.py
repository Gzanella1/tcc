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
from typing import Dict, List

from models.questao import Questao
from utils.text import normalizar_texto, extrair_codigo
from utils.tipo import normalizar_tipo, inferir_tipo


# ─── Mapeamento de aliases de campos ─────────────────────────────────────────
ALIASES_CAMPOS: Dict[str, str] = {
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
    "codigo":               "codigo",
    "código":               "codigo",
    "programa":             "codigo",
    "trecho de codigo":     "codigo",
    "trecho de código":     "codigo",
    "stdin":                "entrada",
    "input":                "entrada",
    "entrada":              "entrada",
    "saida":                "saida",
    "saída":                "saida",
    "output":               "saida",
    "testes":               "testes",
    "casos de teste":       "testes",
}

# Linha separadora de código (ex: ----------------------------------------)
_LINHA_SEPARADORA = re.compile(r"^-{4,}\s*$")


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
            m2 = re.match(r"resposta\s*\d+\s*[-:]\s*(.*)", linha_strip, re.IGNORECASE)
            if m2:
                # Nova questão dentro do mesmo bloco (não deve acontecer, mas protege)
                if m2.group(1).strip():
                    resposta.append(m2.group(1).strip())
                continue
            resposta.append(linha)

    enunciado = normalizar_texto("\n".join(enunciado_linhas))

    return Questao(
        idx=idx,
        tipo=normalizar_tipo(tipo),
        enunciado=enunciado,
        resposta_aluno=normalizar_texto("\n".join(resposta)),
        resposta_referencia="",
        codigo=normalizar_texto("\n".join(codigo)),
        entrada="",
        saida="",
        testes=[],
        extras={},
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

            _campos_conhecidos = {
                "id", "tipo", "enunciado", "pergunta",
                "resposta_aluno", "resposta", "resposta_referencia", "gabarito",
                "codigo", "programa", "entrada", "input", "saida", "output", "testes",
            }

            questoes = []
            for i, item in enumerate(itens, start=1):
                if not isinstance(item, dict):
                    continue

                testes_raw = item.get("testes")
                testes_str = (
                    json.dumps(testes_raw, ensure_ascii=False)
                    if testes_raw is not None
                    else ""
                )

                q = Questao(
                    idx=int(item.get("id", i)),
                    tipo=(
                        normalizar_tipo(str(item.get("tipo", "")))
                        or inferir_tipo(str(item.get("enunciado", item.get("pergunta", ""))))
                    ),
                    enunciado=normalizar_texto(str(item.get("enunciado", item.get("pergunta", "")))),
                    resposta_aluno=normalizar_texto(str(item.get("resposta_aluno", item.get("resposta", "")))),
                    resposta_referencia=normalizar_texto(str(item.get("resposta_referencia", item.get("gabarito", "")))),
                    codigo=normalizar_texto(str(item.get("codigo", item.get("programa", "")))),
                    entrada=normalizar_texto(str(item.get("entrada", item.get("input", "")))),
                    saida=normalizar_texto(str(item.get("saida", item.get("output", "")))),
                    testes=parse_tests_field(testes_str),
                    extras={k: v for k, v in item.items() if k not in _campos_conhecidos},
                )

                if not q.codigo:
                    q.codigo = extrair_codigo(q.enunciado or q.resposta_aluno)

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