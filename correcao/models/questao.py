#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
models/questao.py

Definição dos modelos de dados principais do sistema de correção.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Questao:
    idx: int
    tipo: str = ""
    enunciado: str = ""
    resposta_aluno: str = ""
    resposta_referencia: str = ""
    codigo: str = ""
    entrada: str = ""
    saida: str = ""
    testes: List[Dict[str, str]] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Resultado:
    idx: int
    tipo: str
    nota: float
    status: str
    feedback: str
    detalhes: List[str] = field(default_factory=list)
    testes_executados: List[Dict[str, Any]] = field(default_factory=list)
    saida_correta: str = ""
