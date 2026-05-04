#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation/strategies/previsao.py

Avaliador para questões do tipo PREVISÃO.

Estratégia:
    Executa o código-base da questão e compara a saída real
    com a resposta prevista pelo aluno.
"""

from __future__ import annotations

from config import LIMIAR_APROX, LIMIAR_EXATO
from execution.runner import executar_codigo_python
from models.questao import Questao, Resultado
from utils.text import comparar_textos, extrair_codigo, normalizar_texto


def avaliar(q: Questao) -> Resultado:
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

    entrada  = q.entrada or ""
    execucao = executar_codigo_python(codigo_base, entrada, timeout=3)
    saida_correta = normalizar_texto(execucao["stdout"])

    resposta = q.resposta_aluno or ""
    sim = comparar_textos(resposta, saida_correta)
    ok  = sim >= LIMIAR_APROX

    if execucao["timeout"]:
        status   = "parcial"
        feedback = "O código-base entrou em timeout durante a execução."
        nota     = round(min(4.0, sim * 10), 2)
    elif execucao["erro_execucao"]:
        status   = "parcial"
        feedback = f"Erro ao executar o código-base: {execucao['erro_execucao']}"
        nota     = round(min(5.0, sim * 10), 2)
    elif sim >= LIMIAR_EXATO:
        status   = "ok"
        feedback = "Resposta correta."
        nota     = 10.0
    elif sim >= LIMIAR_APROX:
        status   = "ok"
        feedback = "Resposta muito próxima da saída correta."
        nota     = round(9.0 + (sim - LIMIAR_APROX) * 5, 2)
    else:
        status   = "erro"
        feedback = "Resposta incorreta para a saída esperada."
        nota     = round(sim * 10, 2)

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
                "teste":          1,
                "entrada":        entrada,
                "saida_esperada": saida_correta,
                "saida_obtida":   execucao["stdout"],
                "ok":             ok,
                "motivo":         "ok" if ok else "saída diferente",
                "erro_execucao":  execucao["erro_execucao"],
            }
        ],
        saida_correta=saida_correta,
    )
