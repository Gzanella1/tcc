#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation/dispatcher.py

Roteador principal: seleciona e delega para o avaliador correto
com base no tipo da questão.

Mapeamento:
    previsao     → strategies/previsao.py
    correcao     → strategies/correcao.py
    modificacao  → strategies/modificacao.py
    justificativa→ strategies/justificativa.py
    descritiva   → strategies/descritiva.py
    (fallback)   → tenta código; se falhar, usa texto_llm
"""

from __future__ import annotations

import re

from evaluation.strategies import (
    correcao,
    descritiva,
    justificativa,
    modificacao,
    previsao,
    texto_llm,
)
from evaluation.strategies import codigo as estrategia_codigo
from models.questao import Questao, Resultado
from tests.generator import obter_testes
from utils.text import extrair_codigo, normalizar_texto
from utils.tipo import inferir_tipo, normalizar_tipo
from validation import resposta_correcao_eh_textual, validar_questao


def _resposta_contem_codigo(resposta: str) -> bool:
    """Detecta apenas código explicitamente delimitado na resposta bruta."""
    texto = normalizar_texto(resposta or "")
    return bool(
        "```" in texto
        or re.search(r"(?im)^\s*(?:seu\s+)?c[oó]digo\s*:?\s*$", texto)
    )


def _extrair_resposta_codigo(q: Questao) -> str:
    """Extrai o código ENTREGUE pelo aluno, sem promover texto corrido."""
    if not _resposta_contem_codigo(q.resposta_aluno):
        return ""
    texto  = normalizar_texto(q.resposta_aluno)
    texto  = re.sub(r"(?im)^\s*(?:seu\s+)?c[oó]digo\s*:?\s*$", "", texto).strip()
    codigo = extrair_codigo(texto)
    return codigo.strip() or ""


# Mapa estático: tipo canônico → módulo de estratégia
_ESTRATEGIAS = {
    "previsao":      previsao,
    "correcao":      correcao,
    "modificacao":   modificacao,
    "justificativa": justificativa,
    "descritiva":    descritiva,
}


def corrigir_questao(q: Questao) -> Resultado:
    """
    Seleciona o avaliador adequado e retorna o Resultado da correção.

    Ordem de resolução do tipo:
        1. Campo 'tipo' declarado na questão
        2. Inferência a partir do enunciado
        3. Fallback: tenta código; se nota=0, usa texto via LLM
    """
    tipo = normalizar_tipo(q.tipo) or inferir_tipo(q.enunciado)

    erro_entrada = validar_questao(q)
    if erro_entrada:
        return erro_entrada

    if tipo == "correcao" and resposta_correcao_eh_textual(q):
        return texto_llm.avaliar(q)

    # Rota por tipo conhecido
    estrategia = _ESTRATEGIAS.get(tipo)
    if estrategia:
        return estrategia.avaliar(q)

    # Fallback: questão desconhecida com indícios de código no enunciado.
    if "```" in (q.enunciado or "") or "input(" in (q.enunciado or ""):
        codigo_aluno_resposta = q.codigo_aluno_resposta or _extrair_resposta_codigo(q)
        if not codigo_aluno_resposta:
            return texto_llm.avaliar(q)
        testes       = obter_testes(q)
        res          = estrategia_codigo.avaliar(q, codigo_aluno_resposta, testes)
        if res.nota > 0:
            return res

    # Fallback final: avaliação textual via LLM
    return texto_llm.avaliar(q)
