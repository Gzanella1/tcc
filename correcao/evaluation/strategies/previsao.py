#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation/strategies/previsao.py

Avaliador para questões do tipo PREVISÃO.

Contrato canônico:
    - codigo_base : o programa ANTERIOR do aluno, executado como contexto
      para descobrir a saída que o aluno deve prever. Aqui ele é a
      referência da execução — não é gabarito de código novo e esta
      semântica NÃO se transfere para MODIFICACAO.
    - resposta_aluno : a previsão bruta do aluno.
    - entradaTestes  : entrada de execução OBRIGATÓRIA, produzida
      deterministicamente na GERAÇÃO (dados_entrada → entradaTestes) e
      validada lá por execução de referência. A correção NUNCA deriva a
      entrada do enunciado nem conhece o campo intermediário dados_entrada:
      Questao → entradaTestes → runner → codigo_base → saída → comparação.

Estratégia (em ordem de prioridade):
    1. Tenta extrair pares (entrada → saída prevista) da resposta do aluno,
       pois o aluno frequentemente escreve no formato:
           Entrada: X
           Saída: Y
       (a entrada aqui é a que o PRÓPRIO aluno declara no caso — não vem do
       enunciado nem substitui entradaTestes).
    2. Se não houver pares extraíveis, cai no modo legado: executa o código
       com q.entradaTestes e compara diretamente.

Nota: os prompts de input() (ex: "Digite uma palavra: ") são removidos
da saída antes da comparação, pois o aluno prevê apenas o que os
prints produzem — não os prompts interativos.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from config import LIMIAR_APROX, LIMIAR_EXATO
from evaluation.evidencia import FONTE_AUSENTE, FONTE_EXECUCAO, TIPO_EXECUCAO
from execution.runner import executar_codigo_python
from models.questao import Questao, Resultado
from utils.text import (
    comparar_textos,
    extrair_codigo,
    normalizar_texto,
)


# ─── Extração de pares entrada/saída da resposta do aluno ────────────────────

def _extrair_pares_resposta(resposta: str) -> List[Tuple[str, str]]:
    """
    Tenta extrair pares (entrada, saída_prevista) de respostas no formato:

        Entrada: X
        Saída: Y          ← pode ser multilinha até o próximo "Entrada:" ou fim

    ou variações com acento/sem acento, maiúsculas, dois-pontos, travessão, etc.
    Retorna lista de tuplas (entrada, saida_prevista). Pode ser vazia.

    Estratégia: divide o texto em blocos delimitados por "Entrada: ..."
    e depois busca "Saída: ..." dentro de cada bloco. Isso é mais robusto
    do que um único regex que tenta casar entrada+saída em uma passagem,
    pois funciona com saídas multilinha e conteúdo variado entre os marcadores.
    """
    pares: List[Tuple[str, str]] = []
    texto = resposta.strip()

    # Divide o texto em blocos delimitados por "Entrada: ..."
    blocos = re.split(
        r"(?i)(?:^|\n)\s*(?:entrada|input)\s*[:\-–]\s*",
        texto,
    )

    for bloco in blocos:
        if not bloco.strip():
            continue

        linhas = bloco.splitlines()
        entrada_val = linhas[0].strip()  # primeira linha = valor da entrada
        resto = "\n".join(linhas[1:]).strip()

        # Busca "Saída: ..." no restante do bloco (pode ser multilinha)
        m = re.search(
            r"(?i)(?:sa[íi]da|output|resultado)\s*(?:[:\-–]\s*(?:invertida|resultado|sa[íi]da)?\s*[:\-–]?\s*)(.*)",
            resto,
            re.DOTALL,
        )
        if m:
            saida_val = m.group(1).strip()
            if entrada_val or saida_val:
                pares.append((entrada_val, saida_val))

    return pares


# ─── Remoção de prompts de input() da saída ──────────────────────────────────

def _extrair_prompts_input(codigo: str) -> List[str]:
    """
    Extrai os textos de prompt das chamadas input() no código.
    Ex: input("Digite uma palavra: ") → ["Digite uma palavra: "]
    """
    return re.findall(r'input\s*\(\s*["\']([^"\']*)["\']', codigo)


def _remover_prompts_saida(saida: str, prompts: List[str]) -> str:
    """
    Remove os textos de prompt do input() da saída capturada.
    O Python imprime o prompt do input() junto com o stdout,
    mas o aluno só prevê a saída dos prints — não os prompts.
    """
    if not prompts:
        return saida
    for prompt in prompts:
        saida = saida.replace(prompt, "")
    return normalizar_texto(saida)


# ─── Avaliação por pares ──────────────────────────────────────────────────────

def _avaliar_com_pares(
    q: Questao,
    codigo_base: str,
    pares: List[Tuple[str, str]],
) -> Resultado:
    """
    Executa o código-base para cada par (entrada, saída_prevista)
    e calcula a nota pela fração de acertos.
    Remove automaticamente os prompts do input() da saída antes de comparar,
    pois o aluno prevê apenas o que os prints produzem.
    """
    total         = len(pares)
    passou        = 0
    detalhes: list    = []
    testes_exec: list = []
    prompts_input = _extrair_prompts_input(codigo_base)

    for i, (entrada_val, saida_prevista) in enumerate(pares, start=1):
        execucao         = executar_codigo_python(codigo_base, entrada_val, timeout=3)
        saida_real_bruta = normalizar_texto(execucao["stdout"])
        saida_real       = _remover_prompts_saida(saida_real_bruta, prompts_input)
        saida_prev_n     = normalizar_texto(saida_prevista)

        # Previsão é determinística: exige correspondência quase exata.
        # LIMIAR_APROX (0.90) é permissivo demais — aceita saídas erradas
        # com estrutura parecida (ex: "Invertida: a" vs "Invertida: cba").
        sim = comparar_textos(saida_prev_n.lower(), saida_real.lower())
        ok  = sim >= LIMIAR_EXATO

        if execucao["timeout"]:
            ok, motivo = False, "timeout"
        elif execucao["erro_execucao"]:
            ok, motivo = False, execucao["erro_execucao"]
        elif ok:
            motivo = "ok"
        else:
            motivo = "saída diferente"

        if ok:
            passou += 1

        detalhes.append(
            f"Caso {i}: {'PASSOU' if ok else 'FALHOU'} "
            f"(entrada={repr(entrada_val)}, similaridade={sim:.3f}, motivo={motivo})"
        )
        testes_exec.append({
            "teste":          i,
            "entrada":        entrada_val,
            "saida_esperada": saida_prev_n,    # o que o aluno previu
            "saida_obtida":   saida_real,      # o que o código realmente produz (gabarito)
            "ok":             ok,
            "motivo":         motivo,
            "erro_execucao":  execucao["erro_execucao"],
        })

    nota = (passou / total) * 10 if total else 0.0

    if passou == total:
        status, feedback = "ok",      "Todas as previsões estão corretas."
    elif passou >= max(1, total // 2):
        status, feedback = "parcial", f"{passou}/{total} previsões corretas."
    else:
        status, feedback = "erro",    f"Apenas {passou}/{total} previsões corretas."

    return Resultado(
        idx=q.idx,
        tipo=q.tipo,
        nota=round(max(0.0, min(10.0, nota)), 2),
        status=status,
        feedback=feedback,
        detalhes=detalhes,
        testes_executados=testes_exec,
        fonte_evidencia=FONTE_EXECUCAO,
        evidencias=[{
            "tipo": TIPO_EXECUCAO,
            "resumo": f"{passou}/{total} previsões verificadas executando o código-base.",
            "dados": {
                "modo": "com_pares",
                "casos_total": total,
                "casos_passaram": passou,
                "referencia": "testes_executados",
            },
        }],
    )


# ─── Modo legado: sem pares extraíveis ───────────────────────────────────────

def _avaliar_modo_legado(q: Questao, codigo_base: str) -> Resultado:
    """
    Fallback: executa o código e compara saída real com a resposta bruta do
    aluno (comportamento original). Remove prompts de input() antes de
    comparar.

    A entrada usada é q.entradaTestes (OBRIGATÓRIO no contrato de PREVISAO).
    A correção nunca deriva a entrada do enunciado nem conhece o campo
    intermediário dados_entrada da geração.
    """
    entrada = normalizar_texto(q.entradaTestes)
    execucao      = executar_codigo_python(codigo_base, entrada, timeout=3)
    prompts_input = _extrair_prompts_input(codigo_base)
    saida_correta = _remover_prompts_saida(
        normalizar_texto(execucao["stdout"]), prompts_input
    )

    resposta = normalizar_texto(q.resposta_aluno or "")
    sim      = comparar_textos(resposta, saida_correta)
    ok       = sim >= LIMIAR_APROX

    if execucao["timeout"]:
        status   = "parcial"
        feedback = "O código-base entrou em timeout durante a execução."
        nota     = round(min(4.0, sim * 10), 2)
    elif execucao["erro_execucao"]:
        status   = "parcial"
        feedback = f"Erro ao executar o código-base: {execucao['erro_execucao']}"
        nota     = round(min(5.0, sim * 10), 2)
    elif sim >= LIMIAR_EXATO:
        status, feedback, nota = "ok",   "Resposta correta.",                        10.0
    elif sim >= LIMIAR_APROX:
        status, feedback, nota = "ok",   "Resposta muito próxima da saída correta.", round(9.0 + (sim - LIMIAR_APROX) * 5, 2)
    else:
        status, feedback, nota = "erro", "Resposta incorreta para a saída esperada.", round(sim * 10, 2)

    return Resultado(
        idx=q.idx,
        tipo=q.tipo,
        nota=max(0.0, min(10.0, nota)),
        status=status,
        feedback=feedback,
        detalhes=[
            "Saída correta calculada a partir do código-base.",
            f"Similaridade com a resposta do aluno: {sim:.3f}",
        ],
        testes_executados=[{
            "teste":          1,
            "entrada":        entrada,
            "saida_esperada": saida_correta,
            "saida_obtida":   execucao["stdout"],
            "ok":             ok,
            "motivo":         "ok" if ok else "saída diferente",
            "erro_execucao":  execucao["erro_execucao"],
        }],
        saida_correta=saida_correta,
        fonte_evidencia=FONTE_EXECUCAO,
        evidencias=[{
            "tipo": TIPO_EXECUCAO,
            "resumo": (
                f"Saída do código-base comparada diretamente com a resposta "
                f"do aluno (similaridade={sim:.3f})."
            ),
            "dados": {
                "modo": "legado",
                "similaridade": round(sim, 3),
                "referencia": "testes_executados",
            },
        }],
    )


# ─── Ponto de entrada principal ───────────────────────────────────────────────

def avaliar(q: Questao) -> Resultado:
    codigo_base = q.codigo_base or extrair_codigo(q.enunciado)

    if not codigo_base:
        return Resultado(
            idx=q.idx,
            tipo=q.tipo,
            nota=0.0,
            status="falha",
            feedback="Não foi possível localizar o código da questão para calcular a saída.",
            detalhes=["Faltou o trecho de código necessário para a previsão."],
            fonte_evidencia=FONTE_AUSENTE,
        )

    resposta = q.resposta_aluno or ""

    # 1. Tenta extrair pares do formato "Entrada: X / Saída: Y" na resposta
    pares = _extrair_pares_resposta(resposta)
    if pares:
        return _avaliar_com_pares(q, codigo_base, pares)

    # 2. Fallback legado: executa com q.entradaTestes e compara diretamente.
    return _avaliar_modo_legado(q, codigo_base)

