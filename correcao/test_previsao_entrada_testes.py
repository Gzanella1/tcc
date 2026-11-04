#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_previsao_entrada_testes.py

Valida o contrato de PREVISAO após a mudança de GERAÇÃO:
a entrada de execução (entradaTestes) é OBRIGATÓRIA e é produzida
deterministicamente na geração (dados_entrada → entradaTestes), validada
lá por execução de referência. A correção nunca deriva a entrada do
enunciado e nunca conhece o campo intermediário dados_entrada.

Cobre (critérios de aceite):
    1. Questao NÃO possui campo dados_entrada (intermediário da geração).
    2. O parser de texto lê a seção "entradaTestes:" exportada pela geração.
    3. JSON com dados_entrada não vaza o campo para Questao/extras.
    4. entradaTestes ausente em PREVISAO = erro_entrada (obrigatório).
    5. A estratégia usa SOMENTE q.entradaTestes como entrada de execução.
    6. Caso dicionário {"a":5,"b":3,"c":5}, busca "5" → saída ['a', 'c'].
    7. PREVISAO é objetiva (execução): NÃO chama o LLM.
    8. Regressão: demais tipos não mudam de comportamento.

Os testes de execução real usam o container (mesma convenção de
test_evidencia_propagacao). Docker/podman deve estar disponível.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evaluation.dispatcher import corrigir_questao
from evaluation.evidencia import FONTE_EXECUCAO
from evaluation.strategies import previsao as estrategia_previsao
from models.questao import Questao
from parsing.parser import carregar_questoes
from utils.text import normalizar_texto
from validation import STATUS_ERRO_ENTRADA, validar_questao


CODIGO_BASE_DICIONARIO = (
    'def busca_por_valores(dicionario, valor_busca):\n'
    '    chaves = []\n'
    '    for chave, valor in dicionario.items():\n'
    '        if valor == valor_busca:\n'
    '            chaves.append(chave)\n'
    '    return chaves\n'
    '\n'
    'dicionario = {}\n'
    'while True:\n'
    '    chave = input("chave:")\n'
    '    if chave == "":\n'
    '        break\n'
    '    valor = input("valor:")\n'
    '    dicionario[chave] = valor\n'
    'valor_busca = input("busca:")\n'
    'print(busca_por_valores(dicionario, valor_busca))\n'
)

ENUNCIADO_DICIONARIO = (
    'O programa abaixo lê pares chave/valor até uma linha em branco e em '
    'seguida um valor de busca. Preveja a saída para o dicionário '
    '{"a": 5, "b": 3, "c": 5} e valor de busca "5".'
)

DADOS_ENTRADA_DICIONARIO = {"dicionario": {"a": 5, "b": 3, "c": 5}, "valor_busca": "5"}
ENTRADA_TESTES_DICIONARIO = "a\n5\nb\n3\nc\n5\n\n5\n"
SAIDA_DICIONARIO = "['a', 'c']"


class PrevisaoContratoSemDadosEntradaTests(unittest.TestCase):
    def test_questao_nao_possui_campo_dados_entrada(self):
        # dados_entrada é intermediário da GERAÇÃO; o contrato da correção
        # (Questao) expõe apenas entradaTestes.
        campos = {f.name for f in dataclasses.fields(Questao)}
        self.assertNotIn("dados_entrada", campos)
        self.assertIn("entradaTestes", campos)

    def test_parser_texto_le_secao_entrada_testes(self):
        texto = (
            "============================================================\n"
            "Exercício gerado com base na sua resposta da questão 1\n"
            "============================================================\n\n"
            f"1 - [PREVISAO] {ENUNCIADO_DICIONARIO}\n\n"
            "Código-base:\n"
            "----------------------------------------\n"
            f"{CODIGO_BASE_DICIONARIO}\n"
            "----------------------------------------\n\n"
            f"entradaTestes: {json.dumps(ENTRADA_TESTES_DICIONARIO)}\n\n"
            "resposta 1 -\n"
            f"{SAIDA_DICIONARIO}\n"
        )
        with tempfile.TemporaryDirectory() as td:
            caminho = Path(td) / "entrada.txt"
            caminho.write_text(texto, encoding="utf-8")
            q = carregar_questoes(caminho)[0]

        self.assertEqual(q.tipo, "previsao")
        self.assertEqual(q.entradaTestes, ENTRADA_TESTES_DICIONARIO)
        self.assertEqual(q.resposta_aluno, SAIDA_DICIONARIO)
        self.assertEqual(q.codigo_aluno_resposta, "")
        self.assertIsNone(validar_questao(q))


class PrevisaoValidacaoTests(unittest.TestCase):
    def test_entrada_ausente_e_erro_de_entrada(self):
        q = Questao(
            idx=1,
            tipo="previsao",
            enunciado=ENUNCIADO_DICIONARIO,
            codigo_base=CODIGO_BASE_DICIONARIO,
            entradaTestes="",
            resposta_aluno=SAIDA_DICIONARIO,
        )

        resultado = validar_questao(q)

        self.assertIsNotNone(resultado)
        self.assertEqual(resultado.status, STATUS_ERRO_ENTRADA)
        self.assertTrue(any("entradaTestes" in d for d in resultado.detalhes))

    def test_entrada_presente_valida(self):
        q = Questao(
            idx=1,
            tipo="previsao",
            enunciado=ENUNCIADO_DICIONARIO,
            codigo_base=CODIGO_BASE_DICIONARIO,
            entradaTestes=ENTRADA_TESTES_DICIONARIO,
            resposta_aluno=SAIDA_DICIONARIO,
        )

        self.assertIsNone(validar_questao(q))


class PrevisaoEstrategiaUsaSomenteEntradaTestesTests(unittest.TestCase):
    """A estratégia usa APENAS q.entradaTestes — nunca o enunciado."""

    def test_entrada_testes_usada_verbatim(self):
        q = Questao(
            idx=1,
            tipo="previsao",
            enunciado=ENUNCIADO_DICIONARIO,
            codigo_base=CODIGO_BASE_DICIONARIO,
            entradaTestes=ENTRADA_TESTES_DICIONARIO,
            resposta_aluno=SAIDA_DICIONARIO,
        )

        with mock.patch(
            "evaluation.strategies.previsao.executar_codigo_python",
            return_value={
                "stdout": "['a', 'c']\n",
                "timeout": False,
                "erro_execucao": "",
            },
        ) as exec_mock:
            res = estrategia_previsao.avaliar(q)

        exec_mock.assert_called_once()
        _, entrada_chamada, *_ = exec_mock.call_args.args
        # A estratégia normaliza a entrada antes de executar (comportamento
        # canônico); o valor vem EXCLUSIVAMENTE de q.entradaTestes.
        self.assertEqual(entrada_chamada, normalizar_texto(ENTRADA_TESTES_DICIONARIO))
        self.assertEqual(res.nota, 10.0)
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.saida_correta, "['a', 'c']")

    def test_enunciado_rico_nao_altera_a_entrada_usada(self):
        # O enunciado cita o dicionário, mas a entrada operacional é
        # exclusivamente entradaTestes (a estratégia não interpreta o texto).
        q = Questao(
            idx=2,
            tipo="previsao",
            enunciado=ENUNCIADO_DICIONARIO,
            codigo_base=CODIGO_BASE_DICIONARIO,
            entradaTestes="3\n",
            resposta_aluno="6",
        )

        with mock.patch(
            "evaluation.strategies.previsao.executar_codigo_python",
            return_value={
                "stdout": "6\n",
                "timeout": False,
                "erro_execucao": "",
            },
        ) as exec_mock:
            res = estrategia_previsao.avaliar(q)

        _, entrada_chamada, *_ = exec_mock.call_args.args
        self.assertEqual(entrada_chamada, "3")


class PrevisaoFluxoCompletoTests(unittest.TestCase):
    """Executa código REAL no container (Docker/Podman)."""

    def test_caso_dicionario_com_busca_produz_lista(self):
        q = Questao(
            idx=1,
            tipo="previsao",
            enunciado=ENUNCIADO_DICIONARIO,
            codigo_base=CODIGO_BASE_DICIONARIO,
            entradaTestes=ENTRADA_TESTES_DICIONARIO,
            resposta_aluno=SAIDA_DICIONARIO,
        )

        self.assertIsNone(validar_questao(q))
        res = corrigir_questao(q)

        self.assertEqual(res.nota, 10.0)
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.saida_correta, SAIDA_DICIONARIO)
        self.assertEqual(res.fonte_evidencia, FONTE_EXECUCAO)
        self.assertEqual(res.testes_executados[0]["entrada"], normalizar_texto(ENTRADA_TESTES_DICIONARIO))
        self.assertEqual(res.testes_executados[0]["erro_execucao"], "")

    def test_fluxo_completo_arquivo_formatado(self):
        texto = (
            "============================================================\n"
            "Exercício gerado com base na sua resposta da questão 3\n"
            "============================================================\n\n"
            f"3 - [PREVISAO] {ENUNCIADO_DICIONARIO}\n\n"
            "Código-base:\n"
            "----------------------------------------\n"
            f"{CODIGO_BASE_DICIONARIO}\n"
            "----------------------------------------\n\n"
            f"entradaTestes: {json.dumps(ENTRADA_TESTES_DICIONARIO)}\n\n"
            "resposta 3 -\n"
            f"{SAIDA_DICIONARIO}\n"
        )
        with tempfile.TemporaryDirectory() as td:
            caminho = Path(td) / "entrada.txt"
            caminho.write_text(texto, encoding="utf-8")
            questoes = carregar_questoes(caminho)
            res = corrigir_questao(questoes[0])

        self.assertEqual(questoes[0].entradaTestes, ENTRADA_TESTES_DICIONARIO)
        self.assertEqual(res.nota, 10.0)
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.saida_correta, SAIDA_DICIONARIO)


class PrevisaoObjetivaTests(unittest.TestCase):
    def test_previsao_objetiva_nao_depende_do_llm(self):
        q = Questao(
            idx=1,
            tipo="previsao",
            enunciado=ENUNCIADO_DICIONARIO,
            codigo_base=CODIGO_BASE_DICIONARIO,
            entradaTestes=ENTRADA_TESTES_DICIONARIO,
            resposta_aluno=SAIDA_DICIONARIO,
        )
        with mock.patch(
            "evaluation.strategies.texto_llm.chamar_llm_json",
            side_effect=AssertionError("PREVISAO não deve chamar o LLM"),
        ):
            res = estrategia_previsao.avaliar(q)
        self.assertEqual(res.nota, 10.0)


class PrevisaoRegressaoOutrosTiposTests(unittest.TestCase):
    def test_outros_tipos_nao_sofrem_regressao(self):
        q_just = Questao(
            idx=1,
            tipo="justificativa",
            enunciado="Justifique o uso de funções.",
            rubrica="Menciona organização.",
            resposta_aluno="Organiza e reutiliza código.",
        )
        self.assertIsNone(validar_questao(q_just))

        q_desc = Questao(
            idx=2,
            tipo="descritiva",
            enunciado="Descreva o código.",
            rubrica="Fluxo condicional.",
            resposta_aluno="Ele avalia condições e imprime.",
        )
        self.assertIsNone(validar_questao(q_desc))

        q_mod = Questao(
            idx=3,
            tipo="modificacao",
            enunciado="Leia dois valores e imprima a soma.",
            codigo_aluno_resposta="a = int(input())\nb = int(input())\nprint(a + b)",
        )
        self.assertIsNone(validar_questao(q_mod))

        q_pre = Questao(
            idx=4,
            tipo="previsao",
            enunciado="Qual será a saída?",
            resposta_aluno="1",
        )
        res = validar_questao(q_pre)
        self.assertIsNotNone(res)
        self.assertEqual(res.status, STATUS_ERRO_ENTRADA)


if __name__ == "__main__":
    unittest.main()