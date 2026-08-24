#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
models/questao.py

Definição dos modelos de dados principais do sistema de correção.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from evaluation.evidencia import FONTE_AUSENTE, normalizar_fonte


@dataclass
class Questao:
    idx: int
    tipo: str = ""
    enunciado: str = ""
    resposta_aluno: str = ""
    resposta_referencia: str = ""
    rubrica: str = ""

    # Contrato novo: separa o codigo fornecido pela questao do codigo
    # entregue pelo aluno como resposta.
    codigo_base: str = ""
    codigo_aluno: str = ""

    # Contrato novo: nome explicito para o oraculo de saida.
    saida_esperada: str = ""

    # Campos legados preservados para compatibilidade com as estrategias atuais.
    codigo: str = ""
    entrada: str = ""
    saida: str = ""
    testes: List[Dict[str, str]] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Mantem compatibilidade entre o contrato novo e os nomes antigos.

        - codigo_base <-> codigo
        - saida_esperada <-> saida
        - codigo_aluno pode preencher resposta_aluno quando a entrada vier
          em JSON estruturado apenas com codigo_aluno.
        """
        if not self.codigo_base and self.codigo:
            self.codigo_base = self.codigo
        if not self.codigo and self.codigo_base:
            self.codigo = self.codigo_base

        if not self.saida_esperada and self.saida:
            self.saida_esperada = self.saida
        if not self.saida and self.saida_esperada:
            self.saida = self.saida_esperada

        if self.codigo_aluno and not self.resposta_aluno:
            self.resposta_aluno = self.codigo_aluno


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

    # Contrato novo (Etapa 4.1): origem da evidência que produziu a nota.
    # Valores válidos definidos em evaluation/evidencia.py:
    #     "execucao" | "llm" | "heuristica" | "ausente"
    # Campo opcional com default FONTE_AUSENTE para não quebrar nenhum
    # caminho existente que cria Resultado sem informar a fonte.
    fonte_evidencia: str = FONTE_AUSENTE

    # Contrato novo (Etapa 4.3): evidências concretas e estruturadas que
    # sustentaram a avaliação (resumo; o detalhe completo permanece nos
    # campos específicos como testes_executados). Cada item é um dicionário
    # serializável {"tipo", "resumo", "dados"[, "peso"]}; tipos válidos em
    # evaluation/evidencia.py: "execucao" | "llm" | "heuristica".
    # Ausência de evidência = lista vazia (fonte_evidencia == "ausente").
    # Campo no FINAL do dataclass para preservar construção posicional.
    evidencias: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Garante que a origem da evidência transportada seja sempre um dos
        valores válidos; entradas inválidas degradam para FONTE_AUSENTE.
        """
        self.fonte_evidencia = normalizar_fonte(self.fonte_evidencia)
