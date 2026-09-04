#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_previsao_entrada_enunciado.py

Valida o fluxo completo de PREVISAO quando a entrada NÃO está em
entradaTestes, mas é declarada no próprio enunciado.

Cobre (critérios de aceite):
    1. entradaTestes não precisa estar preenchido (não é erro_entrada).
    2. Entradas presentes no enunciado são reconhecidas (parser e estratégia).
    3. Os números do enunciado NÃO são classificados como codigo_aluno_resposta.
    4. O código-base é executado com as entradas do enunciado.
    5. A saída real é capturada e comparada com a previsão do aluno.
    6. PREVISAO é objetiva (execução), portanto NÃO usa LLM quando a
       execução é suficiente; LLM continua sendo usado nos demais tipos.
    7. A nota deixa de ser 0.0 por erro estrutural (execução com stdin vazio);
       quando a previsão está correta, a nota é 10 e o status é "ok".

Os testes executam código real no container (mesma convenção de
test_evidencia_propagacao). Docker/podman deve estar disponível.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evaluation.dispatcher import corrigir_questao
from evaluation.evidencia import FONTE_EXECUCAO
from evaluation.strategies import previsao as estrategia_previsao
from models.questao import Questao
from parsing.parser import carregar_questoes
from utils.text import extrair_entradas_de_enunciado
from validation import STATUS_ERRO_ENTRADA, validar_questao


CODIGO_BASE_NUMEROS = (
    'lista = []\n'
    'while True:\n'
    '    n = int(input("digite um numero:"))\n'
    '    if (n == 0):\n'
    '        break\n'
    '    else:\n'
    '        lista.append(n)\n'
    '\n'
    'novalista = sorted(lista)\n'
    'for i in novalista:\n'
    '    print(i)'
)

ENUNCIADO_NUMEROS = (
    "Se o usuário digitar os números 5, -3, 0, 1, 0, -1 e depois 0 para "
    "encerrar, qual será a saída esperada do programa?"
)

ENTRADA_NUMEROS = "5\n-3\n0\n1\n0\n-1\n0\n"
SAIDA_NUMEROS = "-3\n5"


class PrevisaoExtraiEntradaNarrativaTests(unittest.TestCase):
    def test_extrai_entradas_de_frases_narrativas(self):
        self.assertEqual(
            extrair_entradas_de_enunciado(ENUNCIADO_NUMEROS, "previsao"),
            ENTRADA_NUMEROS,
        )
        self.assertEqual(
            extrair_entradas_de_enunciado(
                "Se o usuário digitar os números 1, 2, 3 e 4, qual será a saída?",
                "previsao",
            ),
            "1\n2\n3\n4\n",
        )
        self.assertEqual(
            extrair_entradas_de_enunciado(
                "Dado o vetor [2, 3, 4, 5], qual será a saída esperada?",
                "previsao",
            ),
            "2\n3\n4\n5\n",
        )
        self.assertEqual(
            extrair_entradas_de_enunciado("Entrada: 9\nQual a saída?", "previsao"),
            "9\n",
        )

    def test_nao_aplica_narrativa_fora_de_previsao(self):
        self.assertEqual(
            extrair_entradas_de_enunciado(ENUNCIADO_NUMEROS, "modificacao"),
            "",
        )
        # Falso positivo de intervalo não vira entrada.
        self.assertEqual(
            extrair_entradas_de_enunciado(
                "Se o usuário digitar os números entre 1 e 100, qual a saída?",
                "previsao",
            ),
            "",
        )

    def test_parser_nao_trata_numeros_como_codigo_aluno(self):
        texto = (
            f"1 - [PREVISAO] {ENUNCIADO_NUMEROS}\n\n"
            "Código-base:\n"
            "----------------------------------------\n"
            f"{CODIGO_BASE_NUMEROS}\n"
            "----------------------------------------\n\n"
            "resposta 1 -\n"
            "5 -3 0 1 0 -1 \n"
        )
        with tempfile.TemporaryDirectory() as td:
            caminho = Path(td) / "entrada.txt"
            caminho.write_text(texto, encoding="utf-8")
            q = carregar_questoes(caminho)[0]

        self.assertEqual(q.tipo, "previsao")
        self.assertEqual(q.entradaTestes, ENTRADA_NUMEROS)
        self.assertEqual(q.resposta_aluno, "5 -3 0 1 0 -1")
        self.assertEqual(q.codigo_aluno_resposta, "")
        self.assertIsNone(validar_questao(q))


class PrevisaoFluxoCompletoTests(unittest.TestCase):
    def test_questao_sem_entradaTestes_ainda_avalia_ok(self):
        # entradaTestes vazio: nada de erro_entrada; a entrada vem do enunciado.
        q = Questao(
            idx=1,
            tipo="previsao",
            enunciado=ENUNCIADO_NUMEROS,
            codigo_base=CODIGO_BASE_NUMEROS,
            entradaTestes="",
            resposta_aluno=SAIDA_NUMEROS,
        )

        self.assertIsNone(validar_questao(q))
        res = corrigir_questao(q)

        self.assertEqual(res.nota, 10.0)
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.saida_correta, SAIDA_NUMEROS)
        self.assertEqual(res.fonte_evidencia, FONTE_EXECUCAO)
        self.assertEqual(res.testes_executados[0]["entrada"], ENTRADA_NUMEROS)
        self.assertEqual(res.testes_executados[0]["erro_execucao"], "")

    def test_fluxo_completo_arquivo_formatado(self):
        texto = (
            "============================================================\n"
            "Exercício gerado com base na sua resposta da questão 4\n"
            "============================================================\n\n"
            f"4 - [PREVISAO] {ENUNCIADO_NUMEROS}\n\n"
            "Código-base:\n"
            "----------------------------------------\n"
            f"{CODIGO_BASE_NUMEROS}\n"
            "----------------------------------------\n\n"
            "resposta 4 -\n"
            f"{SAIDA_NUMEROS}\n"
        )
        with tempfile.TemporaryDirectory() as td:
            caminho = Path(td) / "entrada.txt"
            caminho.write_text(texto, encoding="utf-8")
            questoes = carregar_questoes(caminho)
            res = corrigir_questao(questoes[0])

        self.assertEqual(questoes[0].entradaTestes, ENTRADA_NUMEROS)
        self.assertEqual(res.nota, 10.0)
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.saida_correta, SAIDA_NUMEROS)

    def test_previsao_incorreta_nao_e_nota_estrutural(self):
        # Reproduz o caso real: a previsão do aluno ("5 -3 0 1 0 -1") está
        # errada. A nota deve vir da comparação honesta, NÃO de uma falha de
        # execução causada por entrada vazia.
        q = Questao(
            idx=1,
            tipo="previsao",
            enunciado=ENUNCIADO_NUMEROS,
            codigo_base=CODIGO_BASE_NUMEROS,
            entradaTestes="",
            resposta_aluno="5 -3 0 1 0 -1",
        )

        res = corrigir_questao(q)

        self.assertNotEqual(res.status, "erro_entrada")
        self.assertEqual(res.fonte_evidencia, FONTE_EXECUCAO)
        self.assertEqual(res.saida_correta, SAIDA_NUMEROS)
        execucao = res.testes_executados[0]
        self.assertEqual(execucao["erro_execucao"], "")
        self.assertEqual(execucao["entrada"], ENTRADA_NUMEROS)
        self.assertIn("Resposta incorreta", res.feedback)

    def test_previsao_objetiva_nao_depende_do_llm(self):
        q = Questao(
            idx=1,
            tipo="previsao",
            enunciado=ENUNCIADO_NUMEROS,
            codigo_base=CODIGO_BASE_NUMEROS,
            resposta_aluno=SAIDA_NUMEROS,
        )
        with mock.patch(
            "evaluation.strategies.texto_llm.chamar_llm_json",
            side_effect=AssertionError("PREVISAO não deve chamar o LLM"),
        ):
            res = estrategia_previsao.avaliar(q)
        self.assertEqual(res.nota, 10.0)


class PrevisaoRegressaoFormatosTests(unittest.TestCase):
    def test_entradaTestes_explicito_continua_funcionando(self):
        q = Questao(
            idx=2,
            tipo="previsao",
            enunciado="Qual será a saída?",
            codigo_base="n = input()\nprint(int(n) * 2)",
            entradaTestes="3\n",
            resposta_aluno="6",
        )
        self.assertIsNone(validar_questao(q))
        res = estrategia_previsao.avaliar(q)
        self.assertEqual(res.nota, 10.0)
        self.assertEqual(res.status, "ok")

    def test_vetor_no_enunciado_continua_funcionando(self):
        q = Questao(
            idx=2,
            tipo="previsao",
            enunciado="Dado o vetor [2, 3, 4, 5], qual será a saída esperada?",
            codigo_base="for _ in range(4):\n    print(int(input()))",
            resposta_aluno="2\n3\n4\n5",
        )
        self.assertIsNone(validar_questao(q))
        res = estrategia_previsao.avaliar(q)
        self.assertEqual(res.nota, 10.0)

    def test_pares_entrada_saida_continua_funcionando(self):
        q = Questao(
            idx=2,
            tipo="previsao",
            enunciado="Qual será a saída?",
            codigo_base="n = input()\nprint(n.upper())",
            entradaTestes="abc\n",
            resposta_aluno="Entrada: abc\nSaída: ABC",
        )
        res = estrategia_previsao.avaliar(q)
        self.assertEqual(res.nota, 10.0)
        self.assertEqual(res.status, "ok")

    def test_outros_tipos_nao_sofrem_regressao(self):
        # justificativa completa → válida
        q_just = Questao(
            idx=1,
            tipo="justificativa",
            enunciado="Justifique o uso de funções.",
            rubrica="Menciona organização.",
            resposta_aluno="Organiza e reutiliza código.",
        )
        self.assertIsNone(validar_questao(q_just))

        # descritiva completa → válida
        q_desc = Questao(
            idx=2,
            tipo="descritiva",
            enunciado="Descreva o código.",
            rubrica="Fluxo condicional.",
            resposta_aluno="Ele avalia condições e imprime.",
        )
        self.assertIsNone(validar_questao(q_desc))

        # modificacao com código e interface com input → válida
        q_mod = Questao(
            idx=3,
            tipo="modificacao",
            enunciado="Leia dois valores e imprima a soma.",
            codigo_aluno_resposta="a = int(input())\nb = int(input())\nprint(a + b)",
        )
        self.assertIsNone(validar_questao(q_mod))

        # previsao sem código-base → erro de entrada
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