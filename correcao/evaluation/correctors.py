#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation/correctors.py

Implementações de todos os avaliadores por tipo de questão:
    - avaliar_previsao           → executa o código e compara com a resposta do aluno
    - avaliar_texto_heuristico   → avaliação textual por sobreposição de tokens
    - avaliar_texto_llm          → avaliação textual via LLM
    - avaliar_codigo_por_testes  → executa código do aluno contra casos de teste
    - avaliar_modificacao_com_llm → combina testes + checagem de requisitos via LLM
"""

from __future__ import annotations

from typing import Dict, List

from config import LIMIAR_APROX, LIMIAR_EXATO, USAR_LLM
from execution.runner import (
    executar_codigo_python,
    executar_codigo_python_sem_entrada,
    verificar_sintaxe_python,
)
from llm.client import chamar_llm_json
from models.questao import Questao, Resultado
from tests.generator import obter_testes
from utils.text import (
    comparar_textos,
    compactar_texto,
    exige_saida_no_enunciado,
    extrair_codigo,
    normalizar_texto,
    sem_acentos,
    tokenizar,
)


# ─── Previsão ────────────────────────────────────────────────────────────────

def avaliar_previsao(q: Questao) -> Resultado:
    """
    Executa o código-base da questão e compara a saída real
    com a resposta prevista pelo aluno.
    """
    codigo_base = q.codigo or extrair_codigo(q.enunciado)
    if not codigo_base:
        return Resultado(
            idx=q.idx,
            tipo=q.tipo,
            nota=0.0,
            status="falha",
            feedback="Não foi possível localizar o código da questão para calcular a saída.",
            detalhes=["Faltou o trecho de código necessário para a previsão."],
        )

    entrada  = q.entrada or ""
    execucao = executar_codigo_python(codigo_base, entrada, timeout=3)
    saida_correta = normalizar_texto(execucao["stdout"])

    resposta = q.resposta_aluno or ""
    sim = comparar_textos(resposta, saida_correta)
    ok  = sim >= LIMIAR_APROX

    if execucao["timeout"]:
        status   = "parcial"
        feedback = "O código-base entrou em timeout durante a execução."
        nota     = round(min(4.0, sim * 10), 2)
    elif execucao["erro_execucao"]:
        status   = "parcial"
        feedback = f"Erro ao executar o código-base: {execucao['erro_execucao']}"
        nota     = round(min(5.0, sim * 10), 2)
    elif sim >= LIMIAR_EXATO:
        status   = "ok"
        feedback = "Resposta correta."
        nota     = 10.0
    elif sim >= LIMIAR_APROX:
        status   = "ok"
        feedback = "Resposta muito próxima da saída correta."
        nota     = round(9.0 + (sim - LIMIAR_APROX) * 5, 2)
    else:
        status   = "erro"
        feedback = "Resposta incorreta para a saída esperada."
        nota     = round(sim * 10, 2)

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
        testes_executados=[
            {
                "teste":          1,
                "entrada":        entrada,
                "saida_esperada": saida_correta,
                "saida_obtida":   execucao["stdout"],
                "ok":             ok,
                "motivo":         "ok" if ok else "saída diferente",
                "erro_execucao":  execucao["erro_execucao"],
            }
        ],
        saida_correta=saida_correta,
    )


# ─── Texto heurístico ─────────────────────────────────────────────────────────

def avaliar_texto_heuristico(q: Questao) -> Resultado:
    """
    Avalia respostas textuais por sobreposição de tokens e similaridade
    de sequência — sem usar o LLM.
    """
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

    enunciado        = normalizar_texto(q.enunciado)
    tokens_enunciado = set(tokenizar(enunciado))
    tokens_resposta  = set(tokenizar(resposta))

    if not tokens_enunciado:
        return Resultado(
            idx=q.idx,
            tipo=q.tipo,
            nota=5.0,
            status="parcial",
            feedback="Não foi possível usar critérios semânticos completos; correção parcial por fallback.",
            detalhes=["Enunciado muito curto ou ausente."],
        )

    stop = {
        "a", "o", "e", "de", "do", "da", "dos", "das",
        "um", "uma", "uns", "umas", "que", "para", "por",
        "com", "sem", "em", "no", "na", "nos", "nas",
        "ao", "aos", "as", "os", "se", "nao", "não",
        "como", "porque", "qual", "quais", "ser", "sera",
        "será", "foi", "sao", "são", "isso", "isto",
    }

    tokens_enunciado = {t for t in tokens_enunciado if t not in stop and len(t) > 2}
    tokens_resposta  = {t for t in tokens_resposta  if t not in stop and len(t) > 2}

    overlap     = len(tokens_enunciado & tokens_resposta) / max(1, len(tokens_enunciado))
    tamanho     = min(1.0, len(tokens_resposta) / max(8, len(tokens_enunciado)))
    similaridade = comparar_textos(enunciado, resposta)

    nota = (0.45 * overlap + 0.20 * tamanho + 0.35 * similaridade) * 10
    nota = max(0.0, min(10.0, nota))

    if nota >= 8.5:
        status   = "ok"
        feedback = "Resposta boa e coerente com o enunciado."
    elif nota >= 6.0:
        status   = "parcial"
        feedback = "Resposta parcialmente correta, mas ainda pode ser mais precisa."
    else:
        status   = "erro"
        feedback = "Resposta fraca em relação ao pedido do enunciado."

    return Resultado(
        idx=q.idx,
        tipo=q.tipo,
        nota=round(nota, 2),
        status=status,
        feedback=feedback,
        detalhes=[
            f"Sobreposição de termos: {overlap:.3f}",
            f"Fator de tamanho: {tamanho:.3f}",
            f"Similaridade textual: {similaridade:.3f}",
        ],
    )


# ─── Texto via LLM ───────────────────────────────────────────────────────────

def avaliar_texto_llm(q: Questao) -> Resultado:
    """
    Avalia respostas textuais usando o LLM.
    Aplica uma regra objetiva de conceito como piso mínimo de nota
    antes de delegar ao modelo.
    """
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

    enunciado  = normalizar_texto(q.enunciado)
    resp_low   = sem_acentos(resposta.lower())
    enun_low   = sem_acentos(enunciado.lower())

    # ── Regra objetiva de conceito ────────────────────────────────────────────
    conceito_ok = False
    if any(p in enun_low for p in [
        "minusc", "maiusc", "lower", "upper", "vogal", "vogais", "padron"
    ]):
        if any(p in resp_low for p in [
            "minusc", "maiusc", "lower", "upper", "padron", "difer", "compar"
        ]):
            conceito_ok = True

    if conceito_ok:
        nota_minima  = 7.0
        status_base  = "parcial"
        feedback_base = "Há entendimento do conceito principal, mas a explicação pode ser melhor."
    else:
        nota_minima  = 0.0
        status_base  = "erro"
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


# ─── Código por testes ───────────────────────────────────────────────────────

def avaliar_codigo_por_testes(
    q: Questao,
    codigo_aluno: str,
    testes: List[Dict[str, str]],
) -> Resultado:
    """
    Avalia o código do aluno executando-o contra cada caso de teste.
    Fluxo:
        1. Valida se o código não está vazio.
        2. Verifica a sintaxe Python.
        3. Se não há testes, executa sem entrada e aceita se não der erro.
        4. Com testes, executa cada um e compara a saída.
    """
    # 1. Código vazio
    if not codigo_aluno.strip():
        return Resultado(
            idx=q.idx, tipo=q.tipo, nota=0.0, status="erro",
            feedback="Resposta de código vazia.",
            detalhes=["Nenhum código foi fornecido pelo aluno."],
        )

    # 2. Sintaxe
    ok_sintaxe, erro_sintaxe = verificar_sintaxe_python(codigo_aluno)
    if not ok_sintaxe:
        return Resultado(
            idx=q.idx, tipo=q.tipo, nota=0.0, status="erro",
            feedback=f"Erro de sintaxe: {erro_sintaxe}",
            detalhes=["O código não compila em Python."],
        )

    # 3. Sem testes
    if not testes:
        execucao    = executar_codigo_python_sem_entrada(codigo_aluno)
        saida_obtida = normalizar_texto(execucao["stdout"])

        if execucao["timeout"]:
            return Resultado(
                idx=q.idx, tipo=q.tipo, nota=0.0, status="erro",
                feedback="O código entrou em timeout.",
            )
        if execucao["erro_execucao"]:
            return Resultado(
                idx=q.idx, tipo=q.tipo, nota=0.0, status="erro",
                feedback=f"Erro ao executar o código: {execucao['erro_execucao']}",
            )

        return Resultado(
            idx=q.idx, tipo=q.tipo, nota=10.0, status="ok",
            feedback="Código executado corretamente (sem necessidade de testes com input).",
            detalhes=[f"Saída obtida:\n{saida_obtida if saida_obtida else '(vazia)'}"],
        )

    # 4. Com testes
    total    = len(testes)
    passou   = 0
    detalhes = []
    execucoes: List[Dict] = []

    for i, teste in enumerate(testes, start=1):
        entrada           = teste.get("entrada", "")
        saida_esperada    = teste.get("saida",   "")
        obs               = teste.get("obs",     "")

        execucao          = executar_codigo_python(codigo_aluno, entrada, timeout=3)
        saida_obtida      = normalizar_texto(execucao["stdout"])
        saida_esperada_norm = normalizar_texto(saida_esperada)

        sim = comparar_textos(saida_obtida.lower(), saida_esperada_norm.lower())

        if execucao["timeout"]:
            ok     = False
            motivo = "timeout"
        elif execucao["erro_execucao"]:
            ok     = False
            motivo = execucao["erro_execucao"]
        elif sim >= LIMIAR_APROX:
            ok     = True
            motivo = "ok"
        else:
            ok     = False
            motivo = "saída diferente"

        if ok:
            passou += 1

        execucoes.append({
            "teste":          i,
            "entrada":        entrada,
            "saida_esperada": saida_esperada_norm,
            "saida_obtida":   saida_obtida,
            "obs":            obs,
            "ok":             ok,
            "motivo":         motivo,
            "stderr":         normalizar_texto(execucao["stderr"]),
            "returncode":     execucao["returncode"],
            "timeout":        execucao["timeout"],
        })

        detalhes.append(
            f"Teste {i}: {'PASSOU' if ok else 'FALHOU'} "
            f"(similaridade={sim:.3f}, motivo={motivo})"
        )

    # 5. Nota final
    nota = (passou / total) * 10 if total else 0.0

    if passou == total:
        status   = "ok"
        feedback = "Todos os testes passaram."
    elif passou >= max(1, total // 2):
        status   = "parcial"
        feedback = f"{passou}/{total} testes passaram."
    else:
        status   = "erro"
        feedback = f"Apenas {passou}/{total} testes passaram."

    return Resultado(
        idx=q.idx,
        tipo=q.tipo,
        nota=round(max(0.0, min(10.0, nota)), 2),
        status=status,
        feedback=feedback,
        detalhes=detalhes,
        testes_executados=execucoes,
    )


# ─── Modificação com LLM ─────────────────────────────────────────────────────

def avaliar_modificacao_com_llm(q: Questao, codigo_aluno: str) -> Resultado:
    """
    Avalia questões de modificação combinando:
    - execução contra casos de teste (peso 70%)
    - checagem de aderência ao enunciado via LLM (peso 30%)

    Se o enunciado não pedir saída explícita, não penaliza ausência de print/return.
    """
    testes           = obter_testes(q)
    resultado_testes = avaliar_codigo_por_testes(q, codigo_aluno, testes)

    if not USAR_LLM:
        return resultado_testes

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

    obj = chamar_llm_json(
        [
            {"role": "system", "content": "Você corrige modificações de código e devolve JSON válido."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.15,
        max_tokens=1200,
    )

    if isinstance(obj, dict):
        try:
            nota_llm = float(obj.get("nota", 0))
        except Exception:
            nota_llm = 0.0

        status_llm = str(obj.get("status", "parcial")).strip().lower()
        if status_llm not in {"ok", "parcial", "erro"}:
            status_llm = "parcial"

        finais: List[str] = []
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

    return resultado_testes
