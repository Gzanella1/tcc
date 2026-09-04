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
from dataclasses import replace

from execution.runner import executar_codigo_python_sem_entrada
from utils.text import sem_acentos, codigo_tem_input


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
        saida_real = ""

        if q.codigo_base and not codigo_tem_input(q.codigo_base):
            execucao = executar_codigo_python_sem_entrada(
                q.codigo_base,
                timeout=3,
            )

            if (
                not execucao.get("erro_execucao", False)
                and not execucao.get("timeout", False)
            ):
                saida_real = execucao.get("stdout", "")

        q = replace(
            q,
            extras={
                **(q.extras or {}),
                "saida_real_codigo_base": saida_real,
            },
        )

        return avaliar_texto_llm(q)

    testes = obter_testes(q)

    if testes:
        return avaliar_codigo(
            q,
            q.codigo_aluno_resposta,
            testes
        )

    return avaliar_texto_llm(q)