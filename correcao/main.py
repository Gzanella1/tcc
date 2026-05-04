#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py

Ponto de entrada do sistema de correção automática de questões.

Fluxo:
    1. Carrega as questões do arquivo de entrada
    2. Corrige cada questão com o avaliador adequado
    3. Gera e salva o relatório de correção

Variáveis de ambiente relevantes (definidas em config.py):
    ARQUIVO_ENTRADA  → caminho do arquivo de questões
    ARQUIVO_SAIDA    → caminho do relatório de saída
    LLM_BASE_URL     → URL base do servidor LLM
    LLM_MODEL        → modelo a usar
    USAR_LLM         → "0" para desabilitar o LLM
    LLM_TIMEOUT      → timeout em segundos para chamadas ao LLM
"""

from __future__ import annotations

import sys
import traceback

from config import ARQUIVO_ENTRADA, ARQUIVO_SAIDA
from evaluation.dispatcher import corrigir_questao
from parsing.parser import carregar_questoes
from models.questao import Resultado
from report.formatter import gerar_relatorio


def main() -> int:
    # ── Carregamento das questões ─────────────────────────────────────────────
    try:
        questoes = carregar_questoes(ARQUIVO_ENTRADA)
    except Exception as e:
        print(f"Erro ao ler a entrada: {e}", file=sys.stderr)
        return 1

    if not questoes:
        print("Nenhuma questão encontrada no arquivo de entrada.", file=sys.stderr)
        return 1

    # ── Correção ──────────────────────────────────────────────────────────────
    resultados: list[Resultado] = []
    for q in questoes:
        try:
            resultados.append(corrigir_questao(q))
        except Exception as e:
            resultados.append(
                Resultado(
                    idx=q.idx,
                    tipo=q.tipo or "desconhecido",
                    nota=0.0,
                    status="erro",
                    feedback=f"Erro interno ao corrigir a questão: {e}",
                    detalhes=[traceback.format_exc(limit=3)],
                )
            )

    # ── Geração do relatório ──────────────────────────────────────────────────
    relatorio = gerar_relatorio(questoes, resultados)

    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    ARQUIVO_SAIDA.write_text(relatorio, encoding="utf-8")

    media = sum(r.nota for r in resultados) / len(resultados)
    print(f"Correção salva em: {ARQUIVO_SAIDA}")
    print(f"Média geral: {media:.2f}/10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
