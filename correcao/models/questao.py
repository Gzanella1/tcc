#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
models/questao.py

Definição dos modelos de dados principais do sistema de correção.

Contrato canônico de campos (sem aliases, sem sincronização automática):

- enunciado           : a nova tarefa/requisitos que o aluno deve cumprir.
- resposta_aluno      : resposta BRUTA do aluno (texto livre; NÃO assuma que
                        é código e NÃO use como substituto automático de
                        codigo_aluno_resposta).
- resposta_referencia : resposta/orientação de referência do professor.
- rubrica             : critérios de avaliação.
- codigo_base         : EXCLUSIVAMENTE o código ANTERIOR do próprio aluno,
                        usado como contexto/apoio. NÃO é gabarito, não é
                        solução oficial, não é resposta esperada, não é
                        código do professor. Em MODIFICACAO não é corrigido,
                        não é comparado para nota nem gera saída esperada.
- codigo_aluno_resposta : o NOVO código produzido pelo aluno; é o objeto da
                        correção (executável/testável). Nunca pode ser
                        utilizado para gerar a própria régua de avaliação.
- saida_esperada      : oráculo de saída explicitamente declarado. Nunca é
                        auto-preenchido com execução de codigo_base ou com
                        saída da própria resposta.
- entradaTestes       : entrada associada aos testes (substitui "entrada").
- saidaTestes         : saída associada aos testes (substitui "saida");
                        distinta de saida_esperada.
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
    codigo_base: str = ""
    codigo_aluno_resposta: str = ""
    saida_esperada: str = ""
    entradaTestes: str = ""
    saidaTestes: str = ""
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
