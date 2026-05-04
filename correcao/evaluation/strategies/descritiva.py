#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation/strategies/descritiva.py

Avaliador para questões do tipo DESCRITIVA.

Estratégia:
    Avaliação semântica via LLM, priorizando o conceito descrito
    pelo aluno em vez de forma ou gramática.
"""

from __future__ import annotations

from evaluation.strategies.texto_llm import avaliar
from models.questao import Questao, Resultado

# Reexporta diretamente — descritiva usa exatamente a mesma
# estratégia de avaliação textual via LLM que o módulo texto_llm implementa.
__all__ = ["avaliar"]
