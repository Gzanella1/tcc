#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation/evidencia.py

Infraestrutura de rastreabilidade da evidência (Etapa 4.1).

Toda nota produzida pelo sistema deve indicar de onde veio a evidência
utilizada na avaliação. Este módulo é a fonte única de verdade para os
valores válidos dessa origem:

    - execucao   : nota derivada da execução real de código
                   (casos de teste, saída calculada do código-base)
    - llm        : nota produzida por avaliação do LLM
    - heuristica : nota produzida por regra/heurística local,
                   sem LLM e sem execução (ex.: sobreposição de tokens,
                   piso de nota por regra objetiva quando o LLM falha)
    - ausente    : nenhuma evidência foi usada para produzir a nota
                   (erro de entrada, resposta pendente, falha interna,
                   nota zerada sem qualquer análise)

O valor padrão é FONTE_AUSENTE: enquanto um caminho de avaliação não
registrar explicitamente a origem, o Resultado permanece honesto sobre
não ter informação de evidência.
"""

from __future__ import annotations

from typing import Any

FONTE_EXECUCAO = "execucao"
FONTE_LLM = "llm"
FONTE_HEURISTICA = "heuristica"
FONTE_AUSENTE = "ausente"

FONTES_VALIDAS = frozenset(
    {
        FONTE_EXECUCAO,
        FONTE_LLM,
        FONTE_HEURISTICA,
        FONTE_AUSENTE,
    }
)


def normalizar_fonte(valor: Any) -> str:
    """
    Normaliza e valida uma origem de evidência.

    - Aceita variações de caixa e espaços em branco ("  LLM " → "llm").
    - Qualquer valor inválido, vazio ou None é mapeado para FONTE_AUSENTE.

    Garante que o campo transporte SEMPRE um dos valores válidos.
    """
    texto = str(valor if valor is not None else "").strip().lower()
    return texto if texto in FONTES_VALIDAS else FONTE_AUSENTE
