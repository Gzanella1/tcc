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

from evaluation.evidencia import (
    FONTE_AUSENTE,
    FONTE_HEURISTICA,
    FONTE_LLM,
    TIPO_HEURISTICA,
    TIPO_LLM,
)
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
            fonte_evidencia=FONTE_AUSENTE,
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

        # Etapa 4.3: captura a nota original ANTES do piso heurístico,
        # para registrar explicitamente se o piso ajustou o valor.
        nota_original = nota

        status = str(obj.get("status", "parcial")).strip().lower()
        if status not in {"ok", "parcial", "erro"}:
            status = "parcial"

        acertos   = obj.get("acertos",   [])
        melhorias = obj.get("melhorias", [])
        if not isinstance(acertos,   list): acertos   = [str(acertos)]
        if not isinstance(melhorias, list): melhorias = [str(melhorias)]

        piso_aplicado = bool(conceito_ok and nota_original < nota_minima)

        if conceito_ok:
            nota = max(nota, nota_minima)
            if status == "erro":
                status = status_base

        detalhes = []
        if acertos:
            detalhes.append("Acertos: "   + "; ".join(str(x) for x in acertos[:5]))
        if melhorias:
            detalhes.append("Melhorias: " + "; ".join(str(x) for x in melhorias[:5]))

        # Evidência estruturada preserva as listas COMPLETAS (sem truncamento
        # de exibição aplicado aos detalhes) e a intervenção do piso.
        evidencia_llm = {
            "tipo": TIPO_LLM,
            "resumo": (
                f"Avaliação via LLM (nota original {nota_original:.2f}"
                + (f", piso de {nota_minima:.1f} aplicado" if piso_aplicado else "")
                + ")."
            ),
            "dados": {
                "nota_original": nota_original,
                "status": status,
                "acertos": acertos,
                "melhorias": melhorias,
                "piso_aplicado": piso_aplicado,
            },
        }

        return Resultado(
            idx=q.idx,
            tipo=q.tipo,
            nota=max(0.0, min(10.0, round(nota, 2))),
            status=status,
            feedback=str(obj.get("feedback", "")).strip() or feedback_base,
            detalhes=detalhes if detalhes else [feedback_base],
            fonte_evidencia=FONTE_LLM,
            evidencias=[evidencia_llm],
        )

    # ── Fallback quando o LLM não responde ───────────────────────────────────
    evidencia_heuristica = {
        "tipo": TIPO_HEURISTICA,
        "resumo": "Nota definida por regra objetiva local porque o LLM não retornou JSON válido.",
        "dados": {
            "motivo": "llm_sem_json_valido",
            "conceito_ok": conceito_ok,
        },
    }

    if conceito_ok:
        return Resultado(
            idx=q.idx,
            tipo=q.tipo,
            nota=7.5,
            status="parcial",
            feedback=feedback_base,
            detalhes=["Correção feita por regra objetiva porque o LLM não retornou JSON válido."],
            fonte_evidencia=FONTE_HEURISTICA,
            evidencias=[evidencia_heuristica],
        )

    return Resultado(
        idx=q.idx,
        tipo=q.tipo,
        nota=3.0,
        status=status_base,
        feedback=feedback_base,
        detalhes=["Correção feita por fallback heurístico porque o LLM não retornou JSON válido."],
        fonte_evidencia=FONTE_HEURISTICA,
        evidencias=[dict(evidencia_heuristica)],
    )
