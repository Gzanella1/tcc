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

# ─── Execução de código ──────────────────────────────────────────────────────
EXEC_BACKEND = os.getenv("EXEC_BACKEND", "container")
CONTAINER_RUNTIME = os.getenv("CONTAINER_RUNTIME", "auto")
CONTAINER_IMAGE = os.getenv("CONTAINER_IMAGE", "python:3.12-alpine")

EXEC_TIMEOUT_SECONDS = int(os.getenv("EXEC_TIMEOUT_SECONDS", "3"))
EXEC_CPU_SECONDS = int(os.getenv("EXEC_CPU_SECONDS", "3"))
EXEC_MEMORY_MB = int(os.getenv("EXEC_MEMORY_MB", "128"))
EXEC_MAX_PROCESSES = int(os.getenv("EXEC_MAX_PROCESSES", "32"))
EXEC_MAX_OPEN_FILES = int(os.getenv("EXEC_MAX_OPEN_FILES", "64"))
EXEC_MAX_CODE_BYTES = int(os.getenv("EXEC_MAX_CODE_BYTES", "65536"))
EXEC_MAX_STDIN_BYTES = int(os.getenv("EXEC_MAX_STDIN_BYTES", "65536"))
EXEC_MAX_STDOUT_BYTES = int(os.getenv("EXEC_MAX_STDOUT_BYTES", "65536"))
EXEC_MAX_STDERR_BYTES = int(os.getenv("EXEC_MAX_STDERR_BYTES", "65536"))
