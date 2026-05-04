#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation/strategies/justificativa.py

Avaliador para questões do tipo JUSTIFICATIVA.

Estratégia:
    Avaliação semântica via LLM, priorizando o conceito explicado
    pelo aluno em vez de forma ou gramática.
"""

from __future__ import annotations

from evaluation.strategies.texto_llm import avaliar
from models.questao import Questao, Resultado

# Reexporta diretamente — justificativa usa exatamente a mesma
# estratégia de avaliação textual via LLM que o módulo texto_llm implementa.
__all__ = ["avaliar"]
