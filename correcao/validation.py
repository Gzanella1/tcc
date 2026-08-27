#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validation.py

Validacao do contrato de dados de uma Questao antes da correcao.

Contrato canônico:
- MODIFICACAO exige apenas {enunciado, codigo_aluno_resposta}. O campo
  codigo_base é OPCIONAL: sua ausência NÃO configura erro_entrada.
- codigo_base nunca é exigido como régua; nunca é usado para gerar testes
  que avaliem o próprio aluno (apenas para compreender a interface).
"""

from __future__ import annotations

from typing import List, Optional

from models.questao import Questao, Resultado
from utils.text import codigo_tem_input, normalizar_texto, sem_acentos
from utils.tipo import inferir_tipo, normalizar_tipo


STATUS_ERRO_ENTRADA = "erro_entrada"
STATUS_INCONCLUSIVO = "inconclusivo"

# Fase 3 (contrato geração→correção): uma pergunta gerada ainda sem resposta
# do estudante NÃO é erro de entrada nem resposta errada — é estado pendente.
STATUS_PENDENTE = "pendente"

# Campos cuja ausência indica apenas que o estudante ainda não respondeu.
# Todos os demais campos pertencem à própria questão e, se ausentes,
# configuram erro_entrada.
_CAMPOS_DO_ALUNO = {"resposta_aluno", "codigo_aluno_resposta"}


def _tipo(q: Questao) -> str:
    return normalizar_tipo(q.tipo) or inferir_tipo(q.enunciado)


def _tem_testes_objetivos(q: Questao) -> bool:
    """Há régua objetiva declarada (testes, saída de teste ou oráculo)?"""
    return bool(
        q.testes
        or normalizar_texto(q.saida_esperada)
        or normalizar_texto(q.saidaTestes)
    )


def _pode_gerar_testes(q: Questao) -> bool:
    """
    Testes podem ser gerados a partir do CODIGO-BASE (interface do programa).

    O código do aluno (codigo_aluno_resposta) NUNCA participa desta decisão:
    ele não pode gerar a própria régua de avaliação.
    """
    return codigo_tem_input(q.codigo_base)


def _resultado(q: Questao, status: str, feedback: str, detalhes: List[str]) -> Resultado:
    return Resultado(
        idx=q.idx,
        tipo=_tipo(q) or q.tipo or "desconhecido",
        nota=0.0,
        status=status,
        feedback=feedback,
        detalhes=detalhes,
    )


def resposta_correcao_eh_textual(q: Questao) -> bool:
    """
    Decide se uma questao de correcao deve ser tratada como resposta textual.

    A decisao usa primeiro o metadado produzido pelo parser. Se ele nao
    existir, aplica uma heuristica conservadora sobre o enunciado.
    """
    formato = str((q.extras or {}).get("resposta_formato", "")).strip().lower()
    if formato == "texto":
        return True
    if formato == "codigo":
        return False

    if q.codigo_aluno_resposta:
        return False

    enunciado = sem_acentos((q.enunciado or "").lower())
    marcadores_textuais = [
        "qual e o erro",
        "qual e o problema",
        "como corrigir",
        "explique o erro",
        "o que esta errado",
        "por que",
        "justifique",
    ]
    return any(m in enunciado for m in marcadores_textuais)


def validar_questao(q: Questao) -> Optional[Resultado]:
    """
    Retorna None quando a questao possui dados suficientes para seguir.

    Caso contrario, retorna um Resultado explicito que distingue tres
    situacoes (contrato geração→correção):

    - erro_entrada : a QUESTÃO em si está incompleta/malformada;
    - pendente     : a questão é válida, mas o estudante ainda não
                     respondeu (resposta_aluno/codigo_aluno_resposta vazios);
    - inconclusivo : há resposta, mas faltam dados para correção objetiva.

    Nenhum dos casos produz uma nota aparentemente valida.
    """
    tipo = _tipo(q)
    ausentes: List[str] = []
    ausentes_aluno: List[str] = []
    inconclusivos: List[str] = []

    def _registrar(campo: str) -> None:
        if campo in _CAMPOS_DO_ALUNO:
            ausentes_aluno.append(campo)
        else:
            ausentes.append(campo)

    if not normalizar_texto(q.enunciado):
        _registrar("enunciado")

    if not tipo:
        _registrar("tipo")

    if tipo == "previsao":
        if not normalizar_texto(q.codigo_base):
            _registrar("codigo_base")
        if not normalizar_texto(q.entradaTestes):
            _registrar("entradaTestes")
        if not normalizar_texto(q.resposta_aluno):
            _registrar("resposta_aluno")

    elif tipo == "modificacao":
        if not normalizar_texto(q.codigo_aluno_resposta):
            _registrar("codigo_aluno_resposta")
        if (
            not _tem_testes_objetivos(q)
            and not _pode_gerar_testes(q)
            and normalizar_texto(q.codigo_aluno_resposta)
        ):
            inconclusivos.append(
                "questao de modificacao sem testes, saida_esperada ou entrada que permita gerar testes"
            )

    elif tipo == "correcao":
        if not resposta_correcao_eh_textual(q):
            if not normalizar_texto(q.codigo_aluno_resposta):
                _registrar("codigo_aluno_resposta")
            if (
                not _tem_testes_objetivos(q)
                and not _pode_gerar_testes(q)
                and normalizar_texto(q.codigo_aluno_resposta)
            ):
                inconclusivos.append(
                    "questao de correcao de codigo sem testes, saida_esperada ou entrada que permita gerar testes"
                )
        elif not normalizar_texto(q.rubrica or q.resposta_referencia):
            if not normalizar_texto(q.resposta_aluno):
                _registrar("resposta_aluno")
            _registrar("rubrica ou resposta_referencia")
        elif not normalizar_texto(q.resposta_aluno):
            _registrar("resposta_aluno")

    elif tipo in {"justificativa", "descritiva"}:
        if not normalizar_texto(q.resposta_aluno):
            _registrar("resposta_aluno")
        if not normalizar_texto(q.rubrica or q.resposta_referencia):
            _registrar("rubrica ou resposta_referencia")

    if ausentes:
        return _resultado(
            q,
            STATUS_ERRO_ENTRADA,
            "ERRO DE ENTRADA: a questao nao possui todos os campos obrigatorios para este tipo.",
            [f"Campo ausente ou vazio: {campo}" for campo in ausentes],
        )

    # Questão íntegra + sem resposta do aluno → pendente (não é erro).
    if ausentes_aluno:
        return _resultado(
            q,
            STATUS_PENDENTE,
            "PENDENTE: pergunta gerada aguardando resposta do aluno. "
            "Nao ha codigo_aluno_resposta/resposta_aluno ainda; nada foi avaliado.",
            [f"Aguardando resposta do aluno: {campo}" for campo in ausentes_aluno],
        )

    if inconclusivos:
        return _resultado(
            q,
            STATUS_INCONCLUSIVO,
            "INCONCLUSIVO: os dados disponiveis nao permitem uma correcao objetiva confiavel.",
            inconclusivos,
        )

    return None
