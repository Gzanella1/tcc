#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation/strategies/correcao.py

Avaliador para questões do tipo CORREÇÃO.

Estratégia:
    - Se o enunciado pede uma explicação textual ("qual é o erro?", "por que falha?"),
      delega para o avaliador de texto via LLM.
    - Caso contrário, executa o código do aluno contra casos de teste.
    - Se não houver testes disponíveis, também delega para texto via LLM.
"""

from __future__ import annotations

from evaluation.strategies.codigo import avaliar as avaliar_codigo
from evaluation.strategies.texto_llm import avaliar as avaliar_texto_llm
from models.questao import Questao, Resultado
from tests.generator import obter_testes
from utils.text import extrair_codigo, normalizar_texto, sem_acentos
import re


def _pergunta_eh_textual(enunciado: str) -> bool:
    """Detecta se a questão pede explicação textual em vez de código corrigido."""
    e = sem_acentos((enunciado or "").lower())
    return any(p in e for p in [
        "qual e o erro",
        "como corrigir",
        "explique o erro",
        "o que esta errado",
        "por que",
    ])


def _extrair_codigo_resposta(resposta_aluno: str) -> str:
    """Extrai código da resposta do aluno, removendo rótulos e cercas markdown."""
    if not resposta_aluno:
        return ""
    texto  = normalizar_texto(resposta_aluno)
    texto  = re.sub(r"(?im)^\s*c[oó]digo\s*:?\s*$", "", texto).strip()
    codigo = extrair_codigo(texto)
    return codigo.strip() or texto.strip()


def avaliar(q: Questao) -> Resultado:
    if _pergunta_eh_textual(q.enunciado):
        return avaliar_texto_llm(q)

    testes       = obter_testes(q)
    codigo_aluno = _extrair_codigo_resposta(q.resposta_aluno)

    if not testes:
        return avaliar_texto_llm(q)

    return avaliar_codigo(q, codigo_aluno, testes)
