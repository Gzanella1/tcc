#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/generator.py

Geração, validação e deduplicação de casos de teste para questões de código.
Os testes podem vir do próprio enunciado (explícitos) ou ser gerados via LLM.
"""

from __future__ import annotations

from typing import Dict, List

from config import TESTES_ALVO, USAR_LLM
from llm.client import chamar_llm_json
from models.questao import Questao
from utils.text import normalizar_texto, codigo_tem_input


# ─── Deduplicação ────────────────────────────────────────────────────────────

def deduplicar_testes(testes: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove testes com entradas duplicadas, mantendo a primeira ocorrência."""
    vistos: set = set()
    saida: List[Dict[str, str]] = []

    for t in testes:
        entrada = normalizar_texto(str(t.get("entrada", "")))
        if entrada in vistos:
            continue
        vistos.add(entrada)
        saida.append({
            "entrada": entrada,
            "saida":   normalizar_texto(str(t.get("saida", ""))),
            "obs":     normalizar_texto(str(t.get("obs", ""))),
        })

    return saida


# ─── Validação ───────────────────────────────────────────────────────────────

def validar_testes(testes: List[Dict], requer_input: bool = False) -> List[Dict[str, str]]:
    """
    Filtra testes inválidos.

    Regras:
    - entrada e saida não podem ser None
    - saida não pode ser vazia
    - se o código usa input(), entrada vazia é descartada
    """
    validos = []

    for t in testes:
        if not isinstance(t, dict):
            continue

        entrada = t.get("entrada")
        saida = t.get("saida")

        if entrada is None or saida is None:
            continue

        entrada = normalizar_texto(str(entrada))
        saida = normalizar_texto(str(saida))
        obs = normalizar_texto(str(t.get("obs", "")))

        if saida == "":
            continue

        if requer_input and entrada == "":
            continue

        validos.append({
            "entrada": entrada,
            "saida": saida,
            "obs": obs,
        })

    return validos


# ─── Geração via LLM ─────────────────────────────────────────────────────────

def _gerar_testes_llm_once(q: Questao, quantidade: int) -> List[Dict[str, str]]:
    """
    Faz uma única chamada ao LLM pedindo {quantidade} testes para a questão.
    Retorna a lista de testes validados gerada.
    """
    usa_input = codigo_tem_input(q.codigo)

    if usa_input:
        regra_input = "- O código usa input(), então gere entradas realistas com dados para input()."
        regra_entrada = "- entrada deve conter dados válidos e terminar com \\n"
    else:
        regra_input = "- O código NÃO usa input(), então NÃO gere entradas."
        regra_entrada = '- entrada deve ser "" (string vazia)'

    prompt = f"""
Você é um gerador de testes para código Python.

Tipo da questão:
{q.tipo}

IMPORTANTE:
{regra_input}
- A saída deve refletir o comportamento correto do programa conforme o ENUNCIADO
- Se o enunciado pedir uma funcionalidade nova (modificação), considere o programa FINAL correto
- Se o enunciado pedir saída adicional, ela deve aparecer na saída esperada
- Não invente comportamento fora do enunciado
- Respeite rigorosamente o funcionamento real esperado

Enunciado:
{q.enunciado}

Código:
{q.codigo or "(não fornecido)"}

Gere exatamente {quantidade} testes.

Formato obrigatório (JSON puro):
{{
  "testes": [
    {{
      "entrada": "",
      "saida": "saida esperada\\n",
      "obs": "descrição do caso"
    }}
  ]
}}

REGRAS:
{regra_entrada}
- saida deve terminar com \\n
- não escreva nada fora do JSON
"""

    obj = chamar_llm_json(
        [
            {"role": "system", "content": "Você gera testes válidos e retorna apenas JSON puro."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1200,
    )

    testes: List[Dict[str, str]] = []

    if isinstance(obj, dict):
        bruto = obj.get("testes", [])
        if isinstance(bruto, list):
            for item in bruto:
                if isinstance(item, dict):
                    entrada = item.get("entrada", "")
                    saida = item.get("saida", "")

                    if not usa_input:
                        entrada = ""

                    testes.append({
                        "entrada": str(entrada),
                        "saida": str(saida),
                        "obs": str(item.get("obs", "")),
                    })

    return validar_testes(testes, requer_input=usa_input)


def gerar_testes_com_llm(q: Questao, quantidade: int = TESTES_ALVO) -> List[Dict[str, str]]:
    """
    Gera testes via LLM com até 3 tentativas.
    Retorna assim que obtiver ao menos 1 teste válido.
    Caso contrário, retorna o melhor resultado obtido.
    """
    if not USAR_LLM:
        return []

    melhor: List[Dict[str, str]] = []

    for _ in range(3):
        testes = _gerar_testes_llm_once(q, quantidade)

        if len(testes) > len(melhor):
            melhor = testes

        if len(testes) >= 1:
            return testes

    return melhor


# ─── Interface pública ────────────────────────────────────────────────────────

def obter_testes_explicitos(q: Questao) -> List[Dict[str, str]]:
    """Retorna os testes declarados explicitamente no enunciado ou nos campos da questão."""
    testes: List[Dict[str, str]] = []

    entrada = normalizar_texto(str(q.entrada or ""))
    saida = normalizar_texto(str(q.saida or ""))

    if saida:
        if codigo_tem_input(q.codigo):
            if entrada != "":
                testes.append({
                    "entrada": entrada,
                    "saida": saida,
                    "obs": "Caso explícito do enunciado",
                })
        else:
            testes.append({
                "entrada": entrada,
                "saida": saida,
                "obs": "Caso explícito do enunciado",
            })

    if q.testes:
        testes.extend(q.testes)

    return deduplicar_testes(testes)


def obter_testes(q: Questao) -> List[Dict[str, str]]:
    """
    Retorna a lista final de testes para uma questão, combinando:
    - testes explícitos do enunciado
    - testes gerados via LLM (apenas se o código usar input())

    Limita ao máximo definido em TESTES_ALVO.
    """
    testes = obter_testes_explicitos(q)

    if codigo_tem_input(q.codigo):
        gerados = gerar_testes_com_llm(q, quantidade=TESTES_ALVO)
    else:
        gerados = []

    if gerados:
        testes.extend(gerados)

    testes = validar_testes(testes, requer_input=codigo_tem_input(q.codigo))
    testes = deduplicar_testes(testes)

    if not testes:
        return []

    return testes[:TESTES_ALVO]