#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils/text.py

Funções utilitárias para normalização, comparação e tokenização de texto.
"""

from __future__ import annotations

import ast
import difflib
import json
import re
import unicodedata
from typing import Iterable, List


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
    texto = str(texto).replace("\r\n", "\n").replace("\r", "\n")
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
    texto = sem_acentos((texto or "").lower())
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
        r"\bprint(?:e|ar|em|ar)?\b",
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
    if not codigo:
        return False
    return bool(re.search(r"\binput\s*\(", codigo))


def _normalizar_prompt(prompt: str) -> str:
    """Gera variantes razoáveis do prompt para remoção segura."""
    prompt = normalizar_texto(prompt)
    if not prompt:
        return ""
    return prompt


def _unique(seq: Iterable[str]) -> List[str]:
    vistos = set()
    saida: List[str] = []
    for item in seq:
        if not item:
            continue
        if item in vistos:
            continue
        vistos.add(item)
        saida.append(item)
    return saida


def extrair_prompts_input(codigo: str) -> List[str]:
    """
    Extrai apenas os prompts literais passados para input("...").

    A função é propositalmente conservadora:
    - só coleta argumentos literais do input()
    - não tenta adivinhar prompts montados dinamicamente
    - não coleta strings de print(), comentário, etc.
    """
    if not codigo:
        return []

    try:
        arvore = ast.parse(codigo)
    except SyntaxError:
        return []

    prompts: List[str] = []

    for node in ast.walk(arvore):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        if not isinstance(func, ast.Name) or func.id != "input":
            continue

        if not node.args:
            continue

        arg0 = node.args[0]

        if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
            prompts.append(_normalizar_prompt(arg0.value))
            continue

        # Casos simples de f-string literal.
        if isinstance(arg0, ast.JoinedStr):
            partes: List[str] = []
            ok = True
            for parte in arg0.values:
                if isinstance(parte, ast.Constant) and isinstance(parte.value, str):
                    partes.append(parte.value)
                else:
                    ok = False
                    break
            if ok:
                prompts.append(_normalizar_prompt("".join(partes)))

    return _unique([p for p in prompts if p])


def remover_prompts_saida(texto: str, prompts_input: Iterable[str]) -> str:
    """
    Remove da saída apenas os prompts capturados por input().

    Importante:
    - não remove outras mensagens do programa
    - não remove o que vem depois do prompt na mesma linha
    - é útil para comparar saída esperada e saída real quando o LLM
      coloca o texto do input() na resposta esperada
    """
    if texto is None:
        return ""
    texto = normalizar_texto(texto)

    prompts = []
    if isinstance(prompts_input, str):
        prompts = [prompts_input]
    else:
        prompts = list(prompts_input or [])

    prompts = _unique([_normalizar_prompt(p) for p in prompts if p])
    if not prompts:
        return texto

    for prompt in sorted(prompts, key=len, reverse=True):
        variants = _unique([
            prompt,
            prompt.rstrip(),
            prompt.strip(),
        ])
        for variante in variants:
            if not variante:
                continue
            # Remove apenas a ocorrência literal do prompt.
            texto = texto.replace(variante, "")

    # Limpeza leve após a remoção.
    texto = re.sub(r"[ \t]+\n", "\n", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return normalizar_texto(texto)