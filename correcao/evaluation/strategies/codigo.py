#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation/strategies/codigo.py

Avaliador de código Python por execução contra casos de teste.

Usado por:
    - correcao.py   (quando há testes disponíveis)
    - modificacao.py (componente de 70% da nota)

Fluxo:
    1. Rejeita código vazio.
    2. Verifica sintaxe Python.
    3. Se não há testes: executa sem entrada e aceita se não houver erro.
    4. Com testes: executa cada caso e compara saída por similaridade.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from config import LIMIAR_APROX
from execution.runner import (
    executar_codigo_python,
    executar_codigo_python_sem_entrada,
    verificar_sintaxe_python,
)
from models.questao import Questao, Resultado
from utils.text import (
    comparar_textos,
    extrair_prompts_input,
    normalizar_texto,
    remover_prompts_saida,
    saida_contem_esperado,
)


def _detectar_erro_execucao(execucao: Dict) -> Tuple[bool, str]:
    """
    Detecta erro de execução mesmo que o runner não preencha
    corretamente o campo erro_execucao.
    """
    timeout = bool(execucao.get("timeout"))
    erro_exec = str(execucao.get("erro_execucao", "") or "").strip()
    stderr = normalizar_texto(execucao.get("stderr", ""))

    if timeout:
        return True, "timeout"

    if erro_exec:
        return True, erro_exec

    if int(execucao.get("returncode", 0) or 0) != 0:
        if stderr:
            primeira_linha = stderr.splitlines()[0].strip()
            return True, primeira_linha or "erro de execução"
        return True, "erro de execução"

    if stderr:
        primeira_linha = stderr.splitlines()[0].strip()
        if "Traceback" in stderr or "Error" in stderr or "EOFError" in stderr:
            return True, primeira_linha or "erro de execução"

    return False, ""


def avaliar(
    q: Questao,
    codigo_aluno: str,
    testes: List[Dict[str, str]],
) -> Resultado:
    # ── 1. Código vazio ───────────────────────────────────────────────────────
    if not codigo_aluno.strip():
        return Resultado(
            idx=q.idx, tipo=q.tipo, nota=0.0, status="erro",
            feedback="Resposta de código vazia.",
            detalhes=["Nenhum código foi fornecido pelo aluno."],
        )

    # ── 2. Sintaxe ────────────────────────────────────────────────────────────
    ok_sintaxe, erro_sintaxe = verificar_sintaxe_python(codigo_aluno)
    if not ok_sintaxe:
        return Resultado(
            idx=q.idx, tipo=q.tipo, nota=0.0, status="erro",
            feedback=f"Erro de sintaxe: {erro_sintaxe}",
            detalhes=["O código não compila em Python."],
        )

    # ── 3. Sem testes ─────────────────────────────────────────────────────────
    if not testes:
        execucao = executar_codigo_python_sem_entrada(codigo_aluno)
        erro_exec, motivo_exec = _detectar_erro_execucao(execucao)
        saida_obtida = normalizar_texto(execucao["stdout"])

        if erro_exec:
            return Resultado(
                idx=q.idx, tipo=q.tipo, nota=0.0, status="erro",
                feedback=f"Erro ao executar o código: {motivo_exec}",
                detalhes=[normalizar_texto(execucao.get("stderr", ""))],
            )

        return Resultado(
            idx=q.idx, tipo=q.tipo, nota=10.0, status="ok",
            feedback="Código executado corretamente (sem necessidade de testes com input).",
            detalhes=[f"Saída obtida:\n{saida_obtida if saida_obtida else '(vazia)'}"],
        )

    # ── 4. Com testes ─────────────────────────────────────────────────────────
    total = len(testes)
    passou = 0
    detalhes: List[str] = []
    execucoes: List[Dict] = []

    # Extrai prompts do input() do código-base (q.codigo) uma única vez,
    # pois são os mesmos para todos os testes.
    prompts_input = extrair_prompts_input(q.codigo or "")

    for i, teste in enumerate(testes, start=1):
        entrada = teste.get("entrada", "")
        saida_esperada = teste.get("saida", "")
        obs = teste.get("obs", "")

        execucao = executar_codigo_python(codigo_aluno, entrada, timeout=3)

        erro_exec, motivo_exec = _detectar_erro_execucao(execucao)

        # Remove apenas prompts provenientes de input() antes de comparar.
        saida_obtida = remover_prompts_saida(normalizar_texto(execucao["stdout"]), prompts_input)
        saida_esperada_norm = remover_prompts_saida(normalizar_texto(saida_esperada), prompts_input)

        sim = comparar_textos(saida_obtida.lower(), saida_esperada_norm.lower())
        contem_esperado = saida_contem_esperado(saida_obtida, saida_esperada_norm)

        if erro_exec:
            ok, motivo = False, motivo_exec
        elif sim >= LIMIAR_APROX:
            ok, motivo = True, "ok"
        elif contem_esperado:
            ok, motivo = True, "saída esperada presente; há saída extra"
        else:
            ok, motivo = False, "saída diferente"

        if ok:
            passou += 1

        execucoes.append({
            "teste": i,
            "entrada": entrada,
            "saida_esperada": saida_esperada_norm,
            "saida_obtida": saida_obtida,
            "obs": obs,
            "ok": ok,
            "motivo": motivo,
            "stderr": normalizar_texto(execucao["stderr"]),
            "returncode": execucao["returncode"],
            "timeout": execucao["timeout"],
        })

        detalhes.append(
            f"Teste {i}: {'PASSOU' if ok else 'FALHOU'} "
            f"(similaridade={sim:.3f}, motivo={motivo})"
        )

    # ── 5. Nota final ─────────────────────────────────────────────────────────
    nota = (passou / total) * 10 if total else 0.0

    if passou == total:
        status, feedback = "ok", "Todos os testes passaram."
    elif passou >= max(1, total // 2):
        status, feedback = "parcial", f"{passou}/{total} testes passaram."
    else:
        status, feedback = "erro", f"Apenas {passou}/{total} testes passaram."

    return Resultado(
        idx=q.idx,
        tipo=q.tipo,
        nota=round(max(0.0, min(10.0, nota)), 2),
        status=status,
        feedback=feedback,
        detalhes=detalhes,
        testes_executados=execucoes,
    )
