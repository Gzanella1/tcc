#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
report/formatter.py

Formatação e geração do relatório de correção em texto simples.
"""

from __future__ import annotations

import textwrap
from datetime import datetime
from typing import List

from models.questao import Questao, Resultado
from utils.text import normalizar_texto


def formatar_resultado(res: Resultado, q: Questao) -> str:
    """
    Formata o resultado de uma questão como bloco de texto legível,
    incluindo nota, status, feedback, detalhes e execuções de testes.
    """
    linhas: List[str] = []
    linhas.append(f"EXERCÍCIO {res.idx}")
    linhas.append(f"Tipo: {res.tipo}")
    linhas.append(f"Nota: {res.nota:.2f}/10")
    linhas.append(f"Status: {res.status}")
    linhas.append(f"Feedback: {res.feedback}")

    if q.enunciado:
        en = normalizar_texto(q.enunciado)
        if len(en) > 900:
            en = en[:900].rstrip() + "..."
        linhas.append("")
        linhas.append("Enunciado:")
        linhas.append(en)

    if res.saida_correta:
        linhas.append("")
        linhas.append("Saída calculada:")
        linhas.append(res.saida_correta if res.saida_correta else "(vazia)")

    if res.detalhes:
        linhas.append("")
        linhas.append("Detalhes:")
        for d in res.detalhes:
            linhas.append(f"- {d}")

    if res.testes_executados:
        linhas.append("")
        linhas.append("Testes executados:")
        for t in res.testes_executados:
            linhas.append(f"- Teste {t.get('teste', '?')}: {'PASSOU' if t.get('ok') else 'FALHOU'}")
            if t.get("obs"):
                linhas.append(f"  Obs: {t['obs']}")
            if t.get("entrada"):
                linhas.append("  Entrada:")
                linhas.append(textwrap.indent(normalizar_texto(str(t["entrada"])), "    "))
            if t.get("saida_esperada") is not None:
                linhas.append("  Saída esperada:")
                linhas.append(textwrap.indent(normalizar_texto(str(t["saida_esperada"])), "    "))
            if t.get("saida_obtida") is not None:
                linhas.append("  Saída obtida:")
                linhas.append(textwrap.indent(normalizar_texto(str(t["saida_obtida"])), "    "))
            if t.get("motivo"):
                linhas.append(f"  Motivo: {t['motivo']}")
            if t.get("timeout"):
                linhas.append("  Timeout: sim")
            if t.get("stderr"):
                linhas.append("  STDERR:")
                linhas.append(textwrap.indent(normalizar_texto(str(t["stderr"])), "    "))

    linhas.append("\n" + "=" * 72 + "\n")
    return "\n".join(linhas)


def gerar_relatorio(questoes: List[Questao], resultados: List[Resultado]) -> str:
    """
    Gera o relatório completo de correção com resumo estatístico e
    os blocos individuais de cada questão.
    """
    media     = sum(r.nota for r in resultados) / max(1, len(resultados))
    aprovadas = sum(1 for r in resultados if r.nota >= 7.0)

    linhas: List[str] = []
    linhas.append("CORREÇÃO AUTOMÁTICA")
    linhas.append(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    linhas.append(f"Total de questões: {len(resultados)}")
    linhas.append(f"Média geral: {media:.2f}/10")
    linhas.append(f"Questões com nota >= 7.0: {aprovadas}")
    linhas.append("=" * 72)
    linhas.append("")

    for q, r in zip(questoes, resultados):
        linhas.append(formatar_resultado(r, q))

    return "\n".join(linhas).rstrip() + "\n"
