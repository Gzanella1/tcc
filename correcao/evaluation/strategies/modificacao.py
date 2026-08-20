#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation/strategies/modificacao.py

Avaliador para questões do tipo MODIFICAÇÃO.

Estratégia combinada:
    - Se houver testes:
        70% → execução do código do aluno contra casos de teste
        30% → checagem de aderência aos requisitos do enunciado via LLM

    - Se NÃO houver testes:
        usa apenas a avaliação semântica via LLM,
        para evitar que a ausência de testes derrube a nota injustamente.

Regra importante:
    Se o enunciado NÃO pedir saída/retorno explícito, a ausência de
    print() ou return não é penalizada.
"""

from __future__ import annotations

import re
from dataclasses import replace

from config import USAR_LLM
from evaluation.strategies.codigo import avaliar as avaliar_codigo
from llm.client import chamar_llm_json
from models.questao import Questao, Resultado
from tests.generator import obter_testes
from utils.text import (
    codigo_tem_input,
    exige_saida_no_enunciado,
    extrair_codigo,
    normalizar_texto,
)


def _extrair_codigo_resposta(resposta_aluno: str) -> str:
    """
    Extrai o código da resposta do aluno.

    Remove rótulos comuns que aparecem no texto da resposta, como:
        Código:
        Seu código:

    Isso evita que o corretor tente compilar essas linhas como Python.
    """
    if not resposta_aluno:
        return ""

    texto = normalizar_texto(resposta_aluno)

    # Remove rótulos que não fazem parte do código Python.
    texto = re.sub(
        r"(?im)^\s*(?:seu\s+)?c[oó]digo\s*:?\s*$",
        "",
        texto
    ).strip()

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

1. Extraia os REQUISITOS explícitos do enunciado.
2. Para cada requisito:
   - verifique se foi atendido no código do aluno;
   - justifique com base no código, sem inventar comportamento.
3. Identifique:
   - o que foi atendido corretamente;
   - o que está incompleto;
   - o que está incorreto.

=========================
REGRAS IMPORTANTES
=========================

- NÃO invente comportamento que não existe no código.
- NÃO avalie estilo, nome de variável ou formatação.
- Foque APENAS no que o enunciado pede.
- Se o enunciado NÃO pede print/return:
  NÃO penalize ausência de saída.

=========================
CRITÉRIO DE NOTA
=========================

- 10 → todos os requisitos atendidos corretamente.
- 7 a 9 → maioria correta, pequenos problemas.
- 4 a 6 → parcialmente correto.
- 0 a 3 → incorreto ou não atende o requisito principal.

=========================
FORMATO DE SAÍDA
=========================

Retorne APENAS JSON válido neste formato:

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
            {
                "role": "system",
                "content": "Você corrige modificações de código e devolve JSON válido.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.15,
        max_tokens=1200,
    )


def _resultado_sem_testes(q: Questao) -> Resultado:
    """
    Resultado neutro quando não há testes disponíveis.

    Isso evita executar código com input() sem fornecer entrada,
    o que geraria EOFError e produziria um erro falso no relatório.
    """
    return Resultado(
        idx=q.idx,
        tipo=q.tipo,
        nota=0.0,
        status="parcial",
        feedback="Não foram encontrados testes objetivos para execução.",
        detalhes=[
            "A avaliação objetiva por testes não foi aplicada.",
            "A nota será definida pela análise dos requisitos via LLM.",
        ],
        testes_executados=[],
    )


def avaliar(q: Questao) -> Resultado:
    # 1. Extrai o código enviado pelo aluno.
    codigo_aluno = _extrair_codigo_resposta(q.resposta_aluno)

    # 2. Usa o código do aluno como base para detectar input()
    # e gerar testes, caso q.codigo esteja vazio.
    q_para_teste = replace(q, codigo=q.codigo or codigo_aluno)

    # 3. Obtém testes explícitos ou gerados via LLM.
    testes = obter_testes(q_para_teste)

    # 4. Avalia a parte objetiva.
    #
    # Se houver testes, executa normalmente.
    #
    # Se não houver testes e o código tiver input(), NÃO executa sem entrada,
    # pois isso causaria EOFError.
    #
    # Se não houver testes e o código não tiver input(), pode executar sem entrada.
    if testes:
        resultado_testes = avaliar_codigo(q_para_teste, codigo_aluno, testes)
    elif codigo_tem_input(codigo_aluno):
        resultado_testes = _resultado_sem_testes(q)
    else:
        resultado_testes = avaliar_codigo(q_para_teste, codigo_aluno, testes)

    # 5. Se o LLM estiver desativado, retorna apenas a avaliação objetiva.
    if not USAR_LLM:
        return resultado_testes

    # 6. Avalia os requisitos do enunciado via LLM.
    obj = _avaliar_requisitos_llm(q, codigo_aluno)

    # 7. Se o LLM falhar, retorna apenas a avaliação objetiva.
    if not isinstance(obj, dict):
        return resultado_testes

    try:
        nota_llm = float(obj.get("nota", 0))
    except Exception:
        nota_llm = 0.0

    nota_llm = max(0.0, min(10.0, nota_llm))

    status_llm = str(obj.get("status", "parcial")).strip().lower()
    if status_llm not in {"ok", "parcial", "erro"}:
        status_llm = "parcial"

    finais = []

    requisitos_atendidos = obj.get("requisitos_atendidos", [])
    faltantes = obj.get("faltantes", [])

    if isinstance(requisitos_atendidos, list) and requisitos_atendidos:
        finais.append(
            "Requisitos atendidos: "
            + "; ".join(str(x) for x in requisitos_atendidos[:6])
        )

    if isinstance(faltantes, list) and faltantes:
        finais.append(
            "Faltantes: "
            + "; ".join(str(x) for x in faltantes[:6])
        )

    # 8. Calcula a nota final.
    #
    # Com testes:
    #   70% testes + 30% LLM
    #
    # Sem testes:
    #   100% LLM
    #
    # Isso evita o problema:
    #   testes = 0
    #   LLM = 10
    #   nota final = 3
    if testes:
        nota_final = (0.7 * resultado_testes.nota) + (0.3 * nota_llm)
    else:
        nota_final = nota_llm

    # 9. Define o status final.
    if testes:
        if status_llm == "erro" and resultado_testes.status != "ok":
            status_final = "erro"
        elif resultado_testes.status == "ok" and status_llm == "ok":
            status_final = "ok"
        else:
            status_final = "parcial"
    else:
        status_final = status_llm

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