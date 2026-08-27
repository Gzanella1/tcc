#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation/strategies/correcao.py

Avaliador para questões do tipo CORREÇÃO.

Contrato canônico:
    - resposta_aluno        : resposta BRUTA do aluno (usada apenas quando a
                              questão é textual).
    - codigo_aluno_resposta : o novo código do aluno — usado quando a questão
                              é de código. NUNCA é substituído automaticamente
                              por resposta_aluno.

Estratégia:
    - Se o enunciado pede uma explicação textual ("qual é o erro?", "por que falha?"),
      delega para o avaliador de texto via LLM.
    - Caso contrário, executa o codigo_aluno_resposta contra casos de teste.
    - Se não houver testes disponíveis, também delega para texto via LLM.
"""

from __future__ import annotations

from evaluation.strategies.codigo import avaliar as avaliar_codigo
from evaluation.strategies.texto_llm import avaliar as avaliar_texto_llm
from models.questao import Questao, Resultado
from tests.generator import obter_testes
from utils.text import sem_acentos


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


def avaliar(q: Questao) -> Resultado:
    if _pergunta_eh_textual(q.enunciado):
        return avaliar_texto_llm(q)

    testes = obter_testes(q)
    # Objeto da correção: EXCLUSIVAMENTE o novo código do aluno.
    # resposta_aluno (resposta bruta) não é promovido a código aqui —
    # essa extração é responsabilidade exclusiva do parser.
    codigo_aluno_resposta = q.codigo_aluno_resposta or ""

    if not testes:
        return avaliar_texto_llm(q)

    return avaliar_codigo(q, codigo_aluno_resposta, testes)
