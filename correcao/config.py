#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
config.py

Centraliza toda a configuração do sistema via variáveis de ambiente.
"""

from __future__ import annotations

import os
from pathlib import Path

# ─── Arquivos ───────────────────────────────────────────────────────────────
ARQUIVO_ENTRADA = Path(os.getenv("ARQUIVO_ENTRADA", "../conteudo/perguntasGeradas.txt"))
ARQUIVO_SAIDA   = Path(os.getenv("ARQUIVO_SAIDA",   "../conteudo/correcao.txt"))

# ─── LLM ────────────────────────────────────────────────────────────────────
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL    = os.getenv("LLM_MODEL",    "qwen/qwen3-vl-4b")
LLM_TIMEOUT  = int(os.getenv("LLM_TIMEOUT", "90"))
USAR_LLM     = os.getenv("USAR_LLM", "1").strip() not in {"0", "false", "False", "no", "NO"}

# ─── Limiares de similaridade ────────────────────────────────────────────────
LIMIAR_EXATO = 0.99
LIMIAR_APROX = 0.90

# ─── Testes ──────────────────────────────────────────────────────────────────
TESTES_ALVO = 6
