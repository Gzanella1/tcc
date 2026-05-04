#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils/tipo.py

Funções para normalização e inferência do tipo de questão.
"""

from __future__ import annotations

import re

from utils.text import normalizar_label, sem_acentos


def normalizar_tipo(tipo: str) -> str:
    """Normaliza o tipo de questão para um dos valores canônicos."""
    t = normalizar_label(tipo)
    mapa = {
        "correcao":           "correcao",
        "corrigir":           "correcao",
        "correcao de codigo": "correcao",
        "justificativa":      "justificativa",
        "justificar":         "justificativa",
        "descritiva":         "descritiva",
        "descrever":          "descritiva",
        "modificacao":        "modificacao",
        "modificar":          "modificacao",
        "alteracao":          "modificacao",
        "alterar":            "modificacao",
        "previsao":           "previsao",
        "prever":             "previsao",
    }
    return mapa.get(t, t or "")


def inferir_tipo(texto: str) -> str:
    """
    Infere o tipo de questão a partir do texto do enunciado.
    Suporta rótulos explícitos como [MODIFICACAO], [CORRECAO], etc.,
    além de palavras-chave no texto.
    """
    t = sem_acentos((texto or "").lower())

    # Rótulos explícitos entre colchetes
    if re.search(r"\[\s*(modificacao|alteracao)\s*\]", t):
        return "modificacao"
    if re.search(r"\[\s*(correcao|corrigir|corrija|conserte|erro)\s*\]", t):
        return "correcao"
    if re.search(r"\[\s*(previsao|prever)\s*\]", t):
        return "previsao"
    if re.search(r"\[\s*(justificativa|justificar|explique)\s*\]", t):
        return "justificativa"
    if re.search(r"\[\s*(descritiva|descrever)\s*\]", t):
        return "descritiva"

    # Palavras-chave para previsao
    if any(p in t for p in [
        "qual sera a saida",
        "o que imprime",
        "o que sera impresso",
        "preveja a saida",
        "previsao",
        "resultado da execucao",
        "saida do programa",
    ]):
        return "previsao"

    # Palavras-chave para correcao
    if any(p in t for p in [
        "corrija",
        "corrigir",
        "conserte",
        "erro",
        "bug",
        "falha no codigo",
        "correcao",
    ]):
        return "correcao"

    # Palavras-chave para modificacao
    if any(p in t for p in [
        "modifique",
        "modificar",
        "altere",
        "alterar",
        "adapte",
        "reescreva",
        "adicione",
        "incluir",
        "remova",
        "modificacao",
        "alteracao",
    ]):
        return "modificacao"

    # Palavras-chave para justificativa
    if any(p in t for p in [
        "justifique",
        "justificativa",
        "por que",
        "explique por que",
    ]):
        return "justificativa"

    # Palavras-chave para descritiva
    if any(p in t for p in [
        "descreva",
        "descritiva",
        "resuma",
        "explique",
    ]):
        return "descritiva"

    return ""
