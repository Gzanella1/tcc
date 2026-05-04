#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
llm/client.py

Cliente para chamadas ao servidor LLM compatível com a API OpenAI (chat/completions).
Configurado via variáveis de ambiente definidas em config.py.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, List, Optional

from config import LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT, USAR_LLM
from utils.text import extrair_json


def chamar_llm(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 1200,
) -> Optional[str]:
    """
    Envia uma requisição ao endpoint /chat/completions do servidor LLM.
    Retorna o conteúdo textual da resposta, ou None em caso de falha
    ou quando USAR_LLM=false.
    """
    if not USAR_LLM:
        return None

    url = LLM_BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model":       LLM_MODEL,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    dados = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=dados,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        obj = json.loads(raw)
        return obj["choices"][0]["message"]["content"]
    except Exception:
        return None


def chamar_llm_json(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 1200,
) -> Optional[Any]:
    """
    Chama o LLM e tenta fazer parse do resultado como JSON.
    Retorna o objeto Python parseado, ou None em caso de falha.
    """
    texto = chamar_llm(messages, temperature=temperature, max_tokens=max_tokens)
    if texto is None:
        return None
    return extrair_json(texto)
