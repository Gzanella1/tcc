#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils/text.py

Funções utilitárias para normalização, comparação e tokenização de texto.
"""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from typing import List


def sem_acentos(texto: str) -> str:
    """Remove acentos e diacríticos de uma string."""
    texto = unicodedata.normalize("NFD", texto)
    return "".join(ch for ch in texto if unicodedata.category(ch) != "Mn")


def normalizar_label(label: str) -> str:
    """Normaliza um rótulo removendo acentos, espaços extras e caracteres especiais."""
    label = sem_acentos(label).lower().strip()
    label = re.sub(r"[^a-z0-9]+", " ", label)
    return re.sub(r"\s+", " ", label).strip()


def normalizar_texto(texto: str) -> str:
    """Normaliza quebras de linha e remove espaços em branco desnecessários."""
    if texto is None:
        return ""
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    linhas = [ln.rstrip() for ln in texto.split("\n")]
    while linhas and not linhas[0].strip():
        linhas.pop(0)
    while linhas and not linhas[-1].strip():
        linhas.pop()
    return "\n".join(linhas).strip()


def compactar_texto(texto: str) -> str:
    """Remove quebras de linha e espaços extras, retornando texto em linha única."""
    texto = normalizar_texto(texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def tokenizar(texto: str) -> List[str]:
    """Divide o texto em tokens alfanuméricos normalizados."""
    texto = sem_acentos(texto.lower())
    return re.findall(r"[a-z0-9_]+", texto)


def comparar_textos(a: str, b: str) -> float:
    """
    Retorna uma similaridade entre 0 e 1 entre dois textos.
    Considera comparação exata, sem espaços e SequenceMatcher.
    """
    a0 = compactar_texto(a)
    b0 = compactar_texto(b)

    if a0 == b0:
        return 1.0

    a1 = a0.replace(" ", "")
    b1 = b0.replace(" ", "")
    if a1 == b1 and a1:
        return 0.98

    ratio = difflib.SequenceMatcher(None, a0.lower(), b0.lower()).ratio()
    return ratio


def extrair_codigo(texto: str) -> str:
    """
    Extrai código entre cercas markdown.
    Se não houver cercas, retorna o texto bruto.
    """
    if not texto:
        return ""

    blocos = re.findall(
        r"```(?:python|py|c|cpp|java|javascript|txt|text)?\s*\n(.*?)```",
        texto,
        flags=re.S | re.I,
    )
    if blocos:
        return "\n\n".join(bloco.strip() for bloco in blocos if bloco.strip())

    return texto.strip()


def extrair_json(texto: str):
    """
    Tenta extrair e parsear um objeto JSON de um texto livre.
    Remove cercas markdown antes de tentar o parse.
    """
    if not texto:
        return None

    texto = texto.strip()
    texto = re.sub(r"^```[a-zA-Z]*", "", texto)
    texto = re.sub(r"```$", "", texto).strip()

    try:
        return json.loads(texto)
    except Exception:
        pass

    match = re.search(r"\{.*\}", texto, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return None


def exige_saida_no_enunciado(texto: str) -> bool:
    """
    Detecta se o enunciado realmente pede saída na tela / retorno.
    Evita penalizar questões que só pedem modificação estrutural do código.
    """
    t = sem_acentos((texto or "").lower())
    padroes = [
        r"\bprint(?:e|ar|e)?\b",
        r"\bmostre\b",
        r"\bimprima\b",
        r"\bexiba\b",
        r"\bsaida\b",
        r"\bretorne\b",
        r"\breturn\b",
        r"\bdeve retornar\b",
        r"\bmostrar o resultado\b",
    ]
    return any(re.search(p, t) for p in padroes)


def codigo_tem_input(codigo: str) -> bool:
    """Verifica se um trecho de código Python usa a função input()."""
    return "input(" in (codigo or "")
