#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation/strategies/modificacao.py

Avaliador para questões do tipo MODIFICAÇÃO.

Estratégia combinada (ponderada):
    - 70% → execução do código do aluno contra casos de teste
    - 30% → checagem de aderência aos requisitos do enunciado via LLM

Regra importante:
    Se o enunciado NÃO pedir saída/retorno explícito, a ausência de
    print() ou return não é penalizada.
"""

from __future__ import annotations

import re

from config import USAR_LLM
from evaluation.strategies.codigo import avaliar as avaliar_codigo
from llm.client import chamar_llm_json
from models.questao import Questao, Resultado
from tests.generator import obter_testes
from utils.text import exige_saida_no_enunciado, extrair_codigo, normalizar_texto


def _extrair_codigo_resposta(resposta_aluno: str) -> str:
    if not resposta_aluno:
        return ""
    texto  = normalizar_texto(resposta_aluno)
    texto  = re.sub(r"(?im)^\s*c[oó]digo\s*:?\s*$", "", texto).strip()
    codigo = extrair_codigo(texto)
    return codigo.strip() or texto.strip()


def _avaliar_requisitos_llm(q: Questao, codigo_aluno: str) -> dict:
    """Chama o LLM para checar aderência do código do aluno ao enunciado."""
    precisa_saida = exige_saida_no_enunciado(q.enunciado)

    prompt = f"""
        Você é um corretor rigoroso de questões de MODIFICAÇÃO de código Python.

        =========================
        CONTEXTO
        =========================

        Enunciado:
        {q.enunciado}

        O enunciado pede saída/retorno explícito?
        { "SIM" if precisa_saida else "NÃO" }

        Código original:
        {q.codigo or "(não há)"}

        Código do aluno:
        {codigo_aluno or "(vazio)"}

        =========================
        TAREFA
        =========================

        1. Extraia os REQUISITOS explícitos do enunciado
        2. Para cada requisito:
           - verifique se foi atendido no código do aluno
           - justifique com base no código (não invente)
        3. Identifique:
           - o que foi atendido corretamente
           - o que está incompleto
           - o que está incorreto

        =========================
        REGRAS IMPORTANTES
        =========================

        - NÃO invente comportamento que não existe no código
        - NÃO avalie estilo (nome de variável, formatação, etc.)
        - Foque APENAS no que o enunciado pede
        - Se o enunciado NÃO pede print/return:
          NÃO penalize ausência de saída

        =========================
        CRITÉRIO DE NOTA
        =========================

        - 10 → todos os requisitos atendidos corretamente
        - 7 a 9 → maioria correta, pequenos problemas
        - 4 a 6 → parcialmente correto
        - 0 a 3 → incorreto ou não atende o principal

        =========================
        FORMATO DE SAÍDA (JSON)
        =========================

        Retorne APENAS JSON válido:

        {{
          "nota": 0,
          "status": "ok|parcial|erro",
          "cumpre_requisitos": true,
          "requisitos_identificados": ["..."],
          "requisitos_atendidos": ["..."],
          "faltantes": ["..."],
          "feedback": "explicação curta e objetiva"
        }}
        """

    return chamar_llm_json(
        [
            {"role": "system", "content": "Você corrige modificações de código e devolve JSON válido."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.15,
        max_tokens=1200,
    )


def avaliar(q: Questao) -> Resultado:
    testes           = obter_testes(q)
    codigo_aluno     = _extrair_codigo_resposta(q.resposta_aluno)
    resultado_testes = avaliar_codigo(q, codigo_aluno, testes)

    if not USAR_LLM:
        return resultado_testes

    obj = _avaliar_requisitos_llm(q, codigo_aluno)

    if not isinstance(obj, dict):
        return resultado_testes

    try:
        nota_llm = float(obj.get("nota", 0))
    except Exception:
        nota_llm = 0.0

    status_llm = str(obj.get("status", "parcial")).strip().lower()
    if status_llm not in {"ok", "parcial", "erro"}:
        status_llm = "parcial"

    finais = []
    at = obj.get("requisitos_atendidos", [])
    fl = obj.get("faltantes",            [])
    if isinstance(at, list) and at:
        finais.append("Requisitos atendidos: " + "; ".join(str(x) for x in at[:6]))
    if isinstance(fl, list) and fl:
        finais.append("Faltantes: "            + "; ".join(str(x) for x in fl[:6]))

    nota_final = (0.7 * resultado_testes.nota) + (0.3 * max(0.0, min(10.0, nota_llm)))

    if status_llm == "erro" and resultado_testes.status != "ok":
        status_final = "erro"
    elif resultado_testes.status == "ok" and status_llm == "ok":
        status_final = "ok"
    else:
        status_final = "parcial"

    feedback = str(obj.get("feedback", "")).strip() or resultado_testes.feedback

    return Resultado(
        idx=q.idx,
        tipo=q.tipo,
        nota=round(max(0.0, min(10.0, nota_final)), 2),
        status=status_final,
        feedback=feedback,
        detalhes=resultado_testes.detalhes + finais,
        testes_executados=resultado_testes.testes_executados,
    )
