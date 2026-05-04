#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation/strategies/texto_llm.py

Avaliador de respostas textuais via LLM.

Usado diretamente por:
    - justificativa.py
    - descritiva.py
    - correcao.py  (quando a pergunta pede explicação textual)

Estratégia:
    1. Aplica uma regra objetiva de conceito como piso mínimo de nota.
    2. Chama o LLM com o enunciado e a resposta do aluno.
    3. Usa o JSON retornado pelo LLM para extrair nota, status e feedback.
    4. Se o LLM falhar, retorna fallback baseado na regra objetiva.
"""

from __future__ import annotations

from llm.client import chamar_llm_json
from models.questao import Questao, Resultado
from utils.text import normalizar_texto, sem_acentos


def _detectar_conceito_ok(enun_low: str, resp_low: str) -> bool:
    """
    Regra objetiva: verifica se a resposta demonstra o conceito central
    quando o enunciado trata de maiúsculas/minúsculas ou padronização.
    """
    if any(p in enun_low for p in [
        "minusc", "maiusc", "lower", "upper", "vogal", "vogais", "padron"
    ]):
        return any(p in resp_low for p in [
            "minusc", "maiusc", "lower", "upper", "padron", "difer", "compar"
        ])
    return False


def avaliar(q: Questao) -> Resultado:
    resposta = normalizar_texto(q.resposta_aluno)

    if not resposta:
        return Resultado(
            idx=q.idx,
            tipo=q.tipo,
            nota=0.0,
            status="erro",
            feedback="Resposta vazia.",
            detalhes=["Sem resposta para avaliar."],
        )

    enunciado   = normalizar_texto(q.enunciado)
    resp_low    = sem_acentos(resposta.lower())
    enun_low    = sem_acentos(enunciado.lower())
    conceito_ok = _detectar_conceito_ok(enun_low, resp_low)

    if conceito_ok:
        nota_minima   = 7.0
        status_base   = "parcial"
        feedback_base = "Há entendimento do conceito principal, mas a explicação pode ser melhor."
    else:
        nota_minima   = 0.0
        status_base   = "erro"
        feedback_base = "A resposta não demonstra com clareza o conceito pedido."

    prompt = f"""
Você é um corretor de respostas textuais de programação.

Enunciado:
{q.enunciado}

Resposta do aluno:
{q.resposta_aluno or "(vazia)"}

Regras de correção:
- Priorize o CONCEITO, não a gramática.
- Se a ideia principal estiver correta, a nota deve ser alta.
- Erros de português ou frase confusa não devem derrubar muito a nota.
- Se a resposta estiver parcialmente correta, dê nota intermediária.
- Se estiver errada conceitualmente, dê nota baixa.
- Não seja rígido demais com a forma.

Avalie estes itens:
1. entendimento do conceito
2. completude
3. clareza

Retorne APENAS JSON válido neste formato:
{{
  "nota": 0,
  "status": "ok|parcial|erro",
  "feedback": "texto curto",
  "acertos": ["..."],
  "melhorias": ["..."]
}}
"""

    obj = chamar_llm_json(
        [
            {"role": "system", "content": "Você devolve JSON válido e corrige respostas textuais com justiça."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.15,
        max_tokens=900,
    )

    if isinstance(obj, dict):
        try:
            nota = float(obj.get("nota", 0))
        except Exception:
            nota = 0.0

        status = str(obj.get("status", "parcial")).strip().lower()
        if status not in {"ok", "parcial", "erro"}:
            status = "parcial"

        acertos   = obj.get("acertos",   [])
        melhorias = obj.get("melhorias", [])
        if not isinstance(acertos,   list): acertos   = [str(acertos)]
        if not isinstance(melhorias, list): melhorias = [str(melhorias)]

        if conceito_ok:
            nota = max(nota, nota_minima)
            if status == "erro":
                status = status_base

        detalhes = []
        if acertos:
            detalhes.append("Acertos: "   + "; ".join(str(x) for x in acertos[:5]))
        if melhorias:
            detalhes.append("Melhorias: " + "; ".join(str(x) for x in melhorias[:5]))

        return Resultado(
            idx=q.idx,
            tipo=q.tipo,
            nota=max(0.0, min(10.0, round(nota, 2))),
            status=status,
            feedback=str(obj.get("feedback", "")).strip() or feedback_base,
            detalhes=detalhes if detalhes else [feedback_base],
        )

    # ── Fallback quando o LLM não responde ───────────────────────────────────
    if conceito_ok:
        return Resultado(
            idx=q.idx,
            tipo=q.tipo,
            nota=7.5,
            status="parcial",
            feedback=feedback_base,
            detalhes=["Correção feita por regra objetiva porque o LLM não retornou JSON válido."],
        )

    return Resultado(
        idx=q.idx,
        tipo=q.tipo,
        nota=3.0,
        status=status_base,
        feedback=feedback_base,
        detalhes=["Correção feita por fallback heurístico porque o LLM não retornou JSON válido."],
    )
