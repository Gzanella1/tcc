#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/oraculo.py

Oráculo por execução para questões de MODIFICAÇÃO.

Problema que resolve:
    O gerador de testes pede ao LLM a ENTRADA e a SAÍDA esperada de cada
    caso. A saída é "calculada de cabeça" pelo modelo e não é conferida;
    quando ele erra (ex.: "Python Python3 Python" → {'python': 3}), o
    gabarito errado reprova um aluno que acertou.

Solução:
    1. O LLM escreve N soluções de referência INDEPENDENTES do programa já
       modificado, a partir do ENUNCIADO + código-base. O código do aluno
       NUNCA participa (anti-autocircularidade preservada).
    2. Cada entrada gerada é executada em todas as referências válidas.
    3. A saída esperada passa a ser a saída OBSERVADA na execução:
         - referências concordam      → saída executada vira o gabarito;
         - referências discordam      → teste descartado (ambíguo);
         - só uma referência executou → usa a saída dela;
         - nenhuma executou           → teste descartado.
    4. Se nenhuma referência puder ser gerada, os testes voltam inalterados
       (comportamento anterior), para não piorar o caso sem LLM funcional.

A saída prevista originalmente pelo LLM é mantida só para rastreabilidade
(anotada em "obs" quando divergia da execução).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from config import ORACULO_REFERENCIAS
from execution.runner import executar_codigo_python, verificar_sintaxe_python
from llm.client import chamar_llm
from models.questao import Questao
from utils.text import extrair_codigo, normalizar_texto

# Temperaturas distintas para que as referências sejam realmente
# independentes (não a mesma resposta repetida).
_TEMPERATURAS = [0.1, 0.5, 0.8]


def _gerar_solucao_referencia(q: Questao, temperatura: float) -> Optional[str]:
    """Pede ao LLM o programa completo já modificado conforme o enunciado."""
    prompt = f"""
Você é um programador Python experiente.

Abaixo está um programa ORIGINAL e um ENUNCIADO pedindo uma modificação.
Escreva o programa COMPLETO já modificado, implementando EXATAMENTE o que
o enunciado pede — nem mais, nem menos.

Enunciado da modificação:
{q.enunciado}

Programa original:
```python
{q.codigo_base or "(não fornecido)"}
```

REGRAS:
- Mantenha a MESMA leitura de entrada do original: mesmas chamadas input(),
  mesma ordem, mesmos textos de prompt, mesma forma de converter os valores.
- Mantenha o MESMO formato de impressão do original (mesmo print, mesmo tipo
  de estrutura impressa). Mude só o que o enunciado exige.
- Não invente regras que o enunciado não pede (ex.: remover acentos,
  remover números, ignorar pontuação), a menos que o enunciado peça.
- Corrija apenas erros que impediriam o programa de executar
  (ex.: linhas soltas inválidas).
- Responda SOMENTE com um bloco ```python contendo o programa completo.
"""
    texto = chamar_llm(
        [
            {"role": "system", "content": "Você escreve programas Python corretos e completos."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperatura,
        max_tokens=1500,
    )
    if not texto:
        return None

    codigo = extrair_codigo(texto)
    if not codigo.strip():
        return None

    ok, _ = verificar_sintaxe_python(codigo)
    return codigo if ok else None


def gerar_referencias(q: Questao, quantidade: int = ORACULO_REFERENCIAS) -> List[str]:
    """Gera até `quantidade` soluções de referência sintaticamente válidas."""
    referencias: List[str] = []
    for i in range(max(0, quantidade)):
        temperatura = _TEMPERATURAS[i % len(_TEMPERATURAS)]
        codigo = _gerar_solucao_referencia(q, temperatura)
        if codigo:
            referencias.append(codigo)
    return referencias


def _executar_referencia(codigo: str, entrada: str) -> Optional[str]:
    """Saída normalizada da referência, ou None se ela falhou nesta entrada."""
    execucao = executar_codigo_python(codigo, entrada, timeout=3)
    if execucao.get("timeout") or execucao.get("erro_execucao"):
        return None
    if execucao.get("returncode") not in (0, None):
        return None
    saida = normalizar_texto(execucao.get("stdout", ""))
    return saida or None


def aplicar_oraculo(
    testes: List[Dict[str, str]],
    referencias: List[str],
) -> List[Dict[str, str]]:
    """
    Substitui a saída prevista pelo LLM pela saída OBSERVADA executando as
    referências. Testes ambíguos ou não executáveis são descartados.
    """
    resultado: List[Dict[str, str]] = []

    for teste in testes:
        entrada = teste.get("entrada", "")
        saidas = [
            s for s in (_executar_referencia(ref, entrada) for ref in referencias)
            if s is not None
        ]

        if not saidas:
            continue  # nenhuma referência executou → sem gabarito confiável

        if any(s != saidas[0] for s in saidas[1:]):
            continue  # referências discordam → caso ambíguo, não penaliza

        saida_executada = saidas[0]
        saida_prevista = normalizar_texto(teste.get("saida", ""))

        novo = dict(teste)
        novo["saida"] = saida_executada + "\n"

        obs = normalizar_texto(teste.get("obs", ""))
        if saida_prevista and saida_prevista != saida_executada:
            nota_oraculo = (
                f"[gabarito obtido por execução; o LLM havia previsto "
                f"{saida_prevista!r}, descartado]"
            )
            novo["obs"] = f"{obs} {nota_oraculo}".strip()
        resultado.append(novo)

    return resultado


def aplicar_oraculo_modificacao(
    q: Questao,
    testes: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Ponto de entrada: gera as referências e reescreve as saídas dos testes.
    Sem referências válidas, devolve os testes inalterados (fallback).
    """
    if not testes:
        return testes

    referencias = gerar_referencias(q)
    if not referencias:
        return testes

    return aplicar_oraculo(testes, referencias)
