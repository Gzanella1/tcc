#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation/strategies/codigo.py

Avaliador de código Python por execução contra casos de teste.

Usado por:
    - correcao.py   (quando há testes disponíveis)
    - modificacao.py (componente de 70% da nota)

Contrato canônico:
    - codigo_aluno_resposta (parâmetro) é o código executado/avaliado.
    - q.codigo_base fornece apenas a interface (prompts de input()) usada
      na limpeza da saída; nunca é tratado como gabarito.

Fluxo:
    1. Rejeita código vazio.
    2. Verifica sintaxe Python.
    3. Se não há testes: executa sem entrada e aceita se não houver erro.
    4. Com testes: executa cada caso e compara saída por similaridade.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from config import LIMIAR_APROX
from evaluation.evidencia import FONTE_AUSENTE, FONTE_EXECUCAO, TIPO_EXECUCAO
from execution.runner import (
    executar_codigo_python,
    executar_codigo_python_sem_entrada,
    verificar_sintaxe_python,
)
from models.questao import Questao, Resultado
from utils.text import (
    ERRO_EOF_INCOMPATIVEL,
    ERRO_NENHUM,
    ERRO_RUNTIME,
    blocos_na_entrada,
    classificar_erro_execucao,
    comparar_textos,
    contar_inputs_codigo,
    extrair_prompts_input,
    normalizar_texto,
    remover_prompts_saida,
    saida_contem_esperado,
)


def _detectar_erro_execucao(execucao: Dict) -> Tuple[bool, str, str]:
    """
    Detecta erro de execução e classifica seu tipo.

    Retorna (erro_detectado, motivo, classificacao):
        erro_detectado : True se houve erro/timeout
        motivo         : descrição legível do erro
        classificacao  : ERRO_NENHUM | ERRO_RUNTIME | ERRO_EOF_INCOMPATIVEL

    A classificação ERRO_EOF_INCOMPATIVEL indica que o erro é um EOFError,
    potencialmente causado por incompatibilidade entre a entrada fornecida
    e a interface do código. O caller DEVE confirmar a incompatibilidade
    comparando a contagem de input() no código com a quantidade de entradas.

    Se o caller não puder confirmar (ou não quiser tratar incompatibilidade),
    deve tratar ERRO_EOF_INCOMPATIVEL como ERRO_RUNTIME.
    """
    timeout = bool(execucao.get("timeout"))
    erro_exec = str(execucao.get("erro_execucao", "") or "").strip()
    stderr = normalizar_texto(execucao.get("stderr", ""))
    returncode = execucao.get("returncode")

    if timeout:
        return True, "timeout", ERRO_RUNTIME

    # Classificar o erro usando a função utilitária
    classificacao = classificar_erro_execucao(stderr, returncode)

    if classificacao == ERRO_NENHUM and not erro_exec:
        return False, "", ERRO_NENHUM

    # Montar motivo legível
    if erro_exec:
        motivo = erro_exec
    elif int(returncode or 0) != 0:
        if stderr:
            primeira_linha = stderr.splitlines()[0].strip()
            motivo = primeira_linha or "erro de execução"
        else:
            motivo = "erro de execução"
    elif stderr:
        primeira_linha = stderr.splitlines()[0].strip()
        motivo = primeira_linha or "erro de execução"
    else:
        motivo = "erro de execução"

    return True, motivo, classificacao


def avaliar(
    q: Questao,
    codigo_aluno_resposta: str,
    testes: List[Dict[str, str]],
) -> Resultado:
    # ── 1. Código vazio ───────────────────────────────────────────────────────
    if not codigo_aluno_resposta.strip():
        return Resultado(
            idx=q.idx, tipo=q.tipo, nota=0.0, status="erro",
            feedback="Resposta de código vazia.",
            detalhes=["Nenhum código foi fornecido pelo aluno."],
            fonte_evidencia=FONTE_AUSENTE,
        )

    # ── 2. Sintaxe ────────────────────────────────────────────────────────────
    ok_sintaxe, erro_sintaxe = verificar_sintaxe_python(codigo_aluno_resposta)
    if not ok_sintaxe:
        # A nota 0 vem de evidência objetiva: o compilador Python foi
        # executado sobre o código e rejeitou. Fonte = execucao.
        return Resultado(
            idx=q.idx, tipo=q.tipo, nota=0.0, status="erro",
            feedback=f"Erro de sintaxe: {erro_sintaxe}",
            detalhes=["O código não compila em Python."],
            fonte_evidencia=FONTE_EXECUCAO,
            evidencias=[{
                "tipo": TIPO_EXECUCAO,
                "resumo": "Compilador Python executado sobre o código e rejeitou (erro de sintaxe).",
                "dados": {
                    "modo": "sintaxe",
                    "compilou": False,
                    "motivo": erro_sintaxe,
                },
            }],
        )

    # ── 3. Sem testes ─────────────────────────────────────────────────────────
    if not testes:
        execucao = executar_codigo_python_sem_entrada(codigo_aluno_resposta)
        erro_exec, motivo_exec, _ = _detectar_erro_execucao(execucao)
        saida_obtida = normalizar_texto(execucao["stdout"])

        if erro_exec:
            return Resultado(
                idx=q.idx, tipo=q.tipo, nota=0.0, status="erro",
                feedback=f"Erro ao executar o código: {motivo_exec}",
                detalhes=[normalizar_texto(execucao.get("stderr", ""))],
                fonte_evidencia=FONTE_EXECUCAO,
                evidencias=[{
                    "tipo": TIPO_EXECUCAO,
                    "resumo": f"Código executado sem entrada e falhou: {motivo_exec}.",
                    "dados": {
                        "modo": "sem_testes",
                        "erro_execucao": True,
                        "motivo": motivo_exec,
                    },
                }],
            )

        return Resultado(
            idx=q.idx, tipo=q.tipo, nota=10.0, status="ok",
            feedback="Código executado corretamente (sem necessidade de testes com input).",
            detalhes=[f"Saída obtida:\n{saida_obtida if saida_obtida else '(vazia)'}"],
            fonte_evidencia=FONTE_EXECUCAO,
            evidencias=[{
                "tipo": TIPO_EXECUCAO,
                "resumo": "Código executado sem entrada, sem erro.",
                "dados": {
                    "modo": "sem_testes",
                    "erro_execucao": False,
                },
            }],
        )

    # ── 4. Com testes ─────────────────────────────────────────────────────────
    total = len(testes)
    passou = 0
    reprovados = 0
    incompativeis = 0
    detalhes: List[str] = []
    execucoes: List[Dict] = []

    # Extrai prompts do input() do código-base (interface declarada do
    # programa anterior do aluno) uma única vez, pois são os mesmos para
    # todos os testes. codigo_base é contexto de interface — nunca gabarito.
    prompts_input = extrair_prompts_input(q.codigo_base or "")

    for i, teste in enumerate(testes, start=1):
        entrada = teste.get("entrada", "")
        saida_esperada = teste.get("saida", "")
        obs = teste.get("obs", "")

        execucao = executar_codigo_python(codigo_aluno_resposta, entrada, timeout=3)

        erro_exec, motivo_exec, classificacao = _detectar_erro_execucao(execucao)

        # Remove apenas prompts provenientes de input() antes de comparar.
        saida_obtida = remover_prompts_saida(normalizar_texto(execucao["stdout"]), prompts_input)
        saida_esperada_norm = remover_prompts_saida(normalizar_texto(saida_esperada), prompts_input)

        sim = comparar_textos(saida_obtida.lower(), saida_esperada_norm.lower())
        contem_esperado = saida_contem_esperado(saida_obtida, saida_esperada_norm)

        # ── Classificação do teste ───────────────────────────────────────────
        # "aprovado" | "reprovado" | "incompativel"
        compatibilidade = "aprovado"

        if erro_exec:
            # EOFError pode indicar incompatibilidade de interface.
            # Regra conservadora: SOMENTE se o código do aluno tem MAIS
            # chamadas input() do que a quantidade de entradas fornecidas.
            # Isso indica que o teste foi gerado para uma interface com
            # menos inputs do que a interface atual do código do aluno.
            if classificacao == ERRO_EOF_INCOMPATIVEL:
                inputs_codigo = contar_inputs_codigo(codigo_aluno_resposta)
                entradas_teste = blocos_na_entrada(entrada)
                if inputs_codigo > entradas_teste:
                    # Teste incompatível: entrada gerada para interface
                    # antiga, código do aluno tem interface nova com mais
                    # inputs. NÃO reduz a nota.
                    compatibilidade = "incompativel"
                    ok = None  # None = teste não contabiliza
                    motivo = "teste incompatível (entrada não corresponde à interface do código)"
                else:
                    # EOFError mas quantidades compatíveis → erro real
                    ok, motivo = False, motivo_exec
            else:
                # Qualquer outro erro → falha real
                ok, motivo = False, motivo_exec
        elif sim >= LIMIAR_APROX:
            ok, motivo = True, "ok"
        elif contem_esperado:
            ok, motivo = True, "saída esperada presente; há saída extra"
        else:
            ok, motivo = False, "saída diferente"

        if ok is True:
            passou += 1
        elif ok is False:
            reprovados += 1
        else:
            # ok is None → incompatível
            incompativeis += 1

        execucoes.append({
            "teste": i,
            "entrada": entrada,
            "saida_esperada": saida_esperada_norm,
            "saida_obtida": saida_obtida,
            "obs": obs,
            "ok": ok,
            "motivo": motivo,
            "compatibilidade": compatibilidade,
            "stderr": normalizar_texto(execucao["stderr"]),
            "returncode": execucao["returncode"],
            "timeout": execucao["timeout"],
        })

        if compatibilidade == "incompativel":
            detalhes.append(
                f"Teste {i}: INCOMPATÍVEL "
                f"(motivo={motivo})"
            )
        else:
            detalhes.append(
                f"Teste {i}: {'PASSOU' if ok else 'FALHOU'} "
                f"(similaridade={sim:.3f}, motivo={motivo})"
            )

    # ── 5. Nota final ─────────────────────────────────────────────────────────
    # Testes incompatíveis são excluídos do denominador. Se TODOS são
    # incompatíveis, nota = 0 (sem aprovação artificial).
    total_validos = total - incompativeis
    nota = (passou / total_validos) * 10 if total_validos else 0.0

    if total_validos == 0:
        status, feedback = "erro", (
            f"Nenhum teste compatível com a interface do código "
            f"({incompativeis} incompatíveis)."
        )
    elif passou == total_validos:
        status, feedback = "ok", "Todos os testes compatíveis passaram."
    elif passou >= max(1, total_validos // 2):
        status, feedback = "parcial", (
            f"{passou}/{total_validos} testes compatíveis passaram."
            + (f" ({incompativeis} incompatíveis excluídos.)" if incompativeis else "")
        )
    else:
        status, feedback = "erro", (
            f"Apenas {passou}/{total_validos} testes compatíveis passaram."
            + (f" ({incompativeis} incompatíveis excluídos.)" if incompativeis else "")
        )

    # Etapa 4.4: procedência da régua de testes. "_origem" é chave interna do
    # fluxo de geração (tests/generator.py); quando ausente — ex.: testes
    # fornecidos diretamente ao avaliador — o caso é tratado como vindo do
    # enunciado/questão (nunca foi produto da geração por LLM neste fluxo).
    contagem_origem = {"enunciado": 0, "llm": 0}
    for teste in testes:
        origem = teste.get("_origem")
        contagem_origem[origem if origem in contagem_origem else "enunciado"] += 1

    return Resultado(
        idx=q.idx,
        tipo=q.tipo,
        nota=round(max(0.0, min(10.0, nota)), 2),
        status=status,
        feedback=feedback,
        detalhes=detalhes,
        testes_executados=execucoes,
        fonte_evidencia=FONTE_EXECUCAO,
        evidencias=[{
            "tipo": TIPO_EXECUCAO,
            "resumo": f"{passou}/{total_validos} casos de teste compatíveis passaram de {total_validos} válidos ({total} total, {incompativeis} incompatíveis).",
            "dados": {
                "modo": "com_testes",
                "testes_total": total,
                "testes_validos": total_validos,
                "testes_passaram": passou,
                "testes_reprovados": reprovados,
                "testes_incompativeis": incompativeis,
                "testes_por_origem": dict(contagem_origem),
                "referencia": "testes_executados",
            },
        }],
    )
