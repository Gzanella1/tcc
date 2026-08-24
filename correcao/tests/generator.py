#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/generator.py

Geração, validação e deduplicação de casos de teste para questões de código.
Os testes podem vir do próprio enunciado (explícitos) ou ser gerados via LLM.
"""

from __future__ import annotations

import re
from typing import Dict, List

from config import TESTES_ALVO, USAR_LLM
from llm.client import chamar_llm_json
from models.questao import Questao
from utils.text import normalizar_texto, codigo_tem_input, extrair_prompts_input

# ─── Etapa 4.4: procedência da régua de testes ────────────────────────────────
# Chave INTERNA "_origem" nos dicionários de teste; não faz parte da interface
# pública documentada (entrada/saida/obs). Valores válidos: "enunciado" | "llm".
ORIGEM_ENUNCIADO = "enunciado"
ORIGEM_LLM = "llm"
_CHAVE_ORIGEM = "_origem"


def _com_origem(teste: Dict[str, str], origem: str) -> Dict[str, str]:
    """Copia um dicionário de teste adicionando a procedência interna."""
    novo = dict(teste)
    novo[_CHAVE_ORIGEM] = origem
    return novo

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
        novo = {
            "entrada": entrada,
            "saida":   normalizar_texto(str(t.get("saida", ""))),
            "obs":     normalizar_texto(str(t.get("obs", ""))),
        }
        # Etapa 4.4: a procedência acompanha o teste que sobrevive à
        # deduplicação (a regra de qual teste sobrevive permanece a mesma).
        if t.get(_CHAVE_ORIGEM):
            novo[_CHAVE_ORIGEM] = str(t[_CHAVE_ORIGEM])
        saida.append(novo)

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

        novo = {
            "entrada": entrada,
            "saida": saida,
            "obs": obs,
        }
        # Etapa 4.4: preserva a procedência interna na reconstrução.
        if t.get(_CHAVE_ORIGEM):
            novo[_CHAVE_ORIGEM] = str(t[_CHAVE_ORIGEM])
        validos.append(novo)

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
        regra_entrada = ("- entrada deve conter SOMENTE os valores digitados pelo usuário", 
        "um por linha, terminando com \\n. "
        "NÃO inclua textos dos prompts, menus ou mensagens do programa. "
        "Exemplo correto: \"5\\n3\\n1\\n\". "
        "Exemplo errado: \"Digite o primeiro número: 5\\n\".")
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
                        "_origem": ORIGEM_LLM,
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
                    "_origem": ORIGEM_ENUNCIADO,
                })
        else:
            testes.append({
                "entrada": entrada,
                "saida": saida,
                "obs": "Caso explícito do enunciado",
                "_origem": ORIGEM_ENUNCIADO,
            })

    if q.testes:
        # Testes vindos dos próprios dados da questão também são do enunciado.
        # Cópia para não mutar os dicionários originais da Questao.
        testes.extend(
            _com_origem(dict(t), ORIGEM_ENUNCIADO) for t in q.testes
        )

    return deduplicar_testes(testes)


def obter_testes(q: Questao) -> List[Dict[str, str]]:
    """
    Retorna a lista final de testes para uma questão, combinando:
    - testes explícitos do enunciado
    - testes gerados via LLM

    Também corrige entradas interativas geradas incorretamente pelo LLM.
    """
    testes = obter_testes_explicitos(q)

    usa_input = codigo_tem_input(q.codigo)

    if usa_input:
        gerados = gerar_testes_com_llm(q, quantidade=TESTES_ALVO)
    else:
        gerados = []

    if gerados:
        testes.extend(gerados)

    # Corrige entradas do tipo:
    # "Digite o primeiro número: 5"
    # para:
    # "5"
    if usa_input:
        testes_corrigidos = []

        for teste in testes:
            teste_corrigido = dict(teste)
            teste_corrigido["entrada"] = limpar_entrada_interativa(
                teste_corrigido.get("entrada", ""),
                q.codigo,
            )
            testes_corrigidos.append(teste_corrigido)

        testes = testes_corrigidos

    testes = validar_testes(testes, requer_input=usa_input)
    testes = deduplicar_testes(testes)

    if not testes:
        return []

    return testes[:TESTES_ALVO]



def limpar_entrada_interativa(entrada: str, codigo: str) -> str:
    """
    Limpa entradas geradas pelo LLM quando ele inclui textos de prompt.

    Exemplo errado gerado pelo LLM:
        Digite o primeiro número: 5
        Digite o segundo número: 3
        Escolha a operação:
        1 - Soma
        2 - Subtração
        3 - Multiplicação
        4 - Divisão
        Digite a opção desejada: 1

    Entrada correta para stdin:
        5
        3
        1
    """
    entrada = normalizar_texto(entrada)

    if not entrada:
        return ""

    prompts = extrair_prompts_input(codigo)

    linhas = [linha.strip() for linha in entrada.splitlines() if linha.strip()]
    valores = []

    for linha in linhas:
        # Ignora linhas de menu, como:
        # 1 - Soma
        # 2 - Subtração
        if re.match(r"^\d+\s*[-.)]\s*\D+", linha):
            continue

        # Se a linha começa com algum prompt do input(),
        # remove o prompt e mantém apenas o valor digitado.
        removeu_prompt = False

        for prompt in prompts:
            prompt = normalizar_texto(prompt).strip()

            if prompt and linha.startswith(prompt):
                valor = linha[len(prompt):].strip()

                if valor:
                    valores.append(valor)

                removeu_prompt = True
                break

        if removeu_prompt:
            continue

        # Se a linha tem formato "algum texto: valor",
        # pega apenas o valor depois dos dois-pontos.
        if ":" in linha:
            _, direita = linha.rsplit(":", 1)
            valor = direita.strip()

            if valor:
                valores.append(valor)

            continue

        # Caso a linha já seja um valor puro, mantém.
        valores.append(linha)

    if not valores:
        return ""

    return "\n".join(valores) + "\n"