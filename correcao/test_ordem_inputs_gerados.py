#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Regressões para montagem determinística do stdin gerado por LLM.

O LLM pode escolher cenários e valores, mas a ordem final enviada ao
programa deve seguir a interface real extraída do código executado.
"""

from __future__ import annotations

import unittest
from unittest import mock

from models.questao import Questao
from tests.generator import (
    _montar_entrada_por_assinatura,
    limpar_entrada_interativa,
    obter_testes,
)
from utils.text import extrair_assinatura_inputs


class AssinaturaInputsTests(unittest.TestCase):
    def test_extrai_ordem_variaveis_prompts_e_conversores(self):
        codigo = (
            'opcao = int(input("Digite a opção desejada: "))\n'
            'a = int(input("Digite o primeiro número: "))\n'
            'b = float(input("Digite o segundo número: "))\n'
            'nome = input("Nome: ")\n'
        )

        assinatura = extrair_assinatura_inputs(codigo)

        self.assertEqual([i["variavel"] for i in assinatura], ["opcao", "a", "b", "nome"])
        self.assertEqual(
            [i["prompt"] for i in assinatura],
            [
                "Digite a opção desejada:",
                "Digite o primeiro número:",
                "Digite o segundo número:",
                "Nome:",
            ],
        )
        self.assertEqual([i["conversor"] for i in assinatura], ["int", "int", "float", ""])

    def test_fallback_textual_funciona_com_codigo_parcialmente_invalido(self):
        codigo = (
            'opcao = int(input("Opção: "))\n'
            'valor = input("Valor: ")\n'
            "if True print('quebrado')\n"
        )

        assinatura = extrair_assinatura_inputs(codigo)

        self.assertEqual([i["variavel"] for i in assinatura], ["opcao", "valor"])
        self.assertEqual([i["prompt"] for i in assinatura], ["Opção:", "Valor:"])


class MontagemDeterministicaTests(unittest.TestCase):
    def test_caso_1_opcao_primeiro_nao_usa_ordem_errada_do_llm(self):
        codigo = (
            'opcao = int(input("Digite a opção desejada: "))\n'
            'a = int(input("Digite o primeiro número: "))\n'
            'b = int(input("Digite o segundo número: "))\n'
        )
        assinatura = extrair_assinatura_inputs(codigo)
        item = {
            "entrada": "5\n3\n1\n",
            "valores_entrada": {"a": "5", "b": "3", "opcao": "1"},
            "saida": "Soma: 8\n",
        }

        entrada = _montar_entrada_por_assinatura(item, assinatura)

        self.assertEqual(entrada, "1\n5\n3\n")
        self.assertNotEqual(entrada, "5\n3\n1\n")

    def test_caso_2_opcao_depois_dos_valores(self):
        codigo = (
            'a = int(input("Digite o primeiro número: "))\n'
            'b = int(input("Digite o segundo número: "))\n'
            'opcao = int(input("Digite a opção desejada: "))\n'
        )
        assinatura = extrair_assinatura_inputs(codigo)
        item = {
            "valores_entrada": [
                {"variavel": "opcao", "valor": "1"},
                {"variavel": "b", "valor": "3"},
                {"variavel": "a", "valor": "5"},
            ],
            "saida": "Soma: 8\n",
        }

        entrada = _montar_entrada_por_assinatura(item, assinatura)

        self.assertEqual(entrada, "5\n3\n1\n")

    def test_caso_3_codigo_com_apenas_dois_inputs(self):
        codigo = (
            'x = input("X: ")\n'
            'y = input("Y: ")\n'
        )
        assinatura = extrair_assinatura_inputs(codigo)
        item = {"valores_entrada": {"y": "beta", "x": "alfa"}}

        entrada = _montar_entrada_por_assinatura(item, assinatura)

        self.assertEqual(entrada, "alfa\nbeta\n")

    def test_caso_4_varios_inputs_de_tipos_diferentes(self):
        codigo = (
            'nome = input("Nome: ")\n'
            'idade = int(input("Idade: "))\n'
            'altura = float(input("Altura: "))\n'
            'ativo = input("Ativo? ")\n'
        )
        assinatura = extrair_assinatura_inputs(codigo)
        item = {
            "valores_entrada": [
                {"variavel": "altura", "valor": 1.7},
                {"variavel": "ativo", "valor": "sim"},
                {"variavel": "nome", "valor": "Ana"},
                {"variavel": "idade", "valor": 30},
            ]
        }

        entrada = _montar_entrada_por_assinatura(item, assinatura)

        self.assertEqual(entrada, "Ana\n30\n1.7\nsim\n")

    def test_caso_5_mapeia_por_prompts_diferentes(self):
        codigo = (
            'cor = input("Cor favorita? ")\n'
            'comida = input("Comida preferida: ")\n'
        )
        assinatura = extrair_assinatura_inputs(codigo)
        item = {
            "valores_entrada": {
                "Comida preferida:": "lasanha",
                "Cor favorita?": "azul",
            }
        }

        entrada = _montar_entrada_por_assinatura(item, assinatura)

        self.assertEqual(entrada, "azul\nlasanha\n")

    def test_caso_6_limpar_entrada_interativa_nao_reordena_valores(self):
        codigo = (
            'primeiro = input("Primeiro: ")\n'
            'segundo = input("Segundo: ")\n'
        )
        entrada_bruta = "Segundo: b\nPrimeiro: a\n"

        entrada = limpar_entrada_interativa(entrada_bruta, codigo)

        self.assertEqual(entrada, "b\na\n")


class FluxoObterTestesTests(unittest.TestCase):
    @mock.patch("tests.generator.USAR_LLM", True)
    @mock.patch("tests.generator.chamar_llm_json")
    def test_obter_testes_remonta_stdin_gerado_pelo_llm(self, mock_llm):
        mock_llm.return_value = {
            "testes": [
                {
                    "entrada": "5\n3\n1\n",
                    "valores_entrada": {"a": "5", "b": "3", "opcao": "1"},
                    "saida": "Soma: 8\n",
                    "obs": "soma",
                }
            ]
        }
        q = Questao(
            idx=1,
            tipo="modificacao",
            enunciado="Adicione um menu de operações. Para opção 1, some a e b.",
            codigo_base="a = int(input())\nb = int(input())\nprint(a + b)",
            codigo_aluno_resposta=(
                'opcao = int(input("Digite a opção desejada: "))\n'
                'a = int(input("Digite o primeiro número: "))\n'
                'b = int(input("Digite o segundo número: "))\n'
                'if opcao == 1:\n'
                '    print(f"Soma: {a + b}")\n'
            ),
        )

        testes = obter_testes(q)

        self.assertEqual(testes[0]["entrada"], "1\n5\n3")
        self.assertNotEqual(testes[0]["entrada"], "5\n3\n1")

    @mock.patch("tests.generator.USAR_LLM", False)
    def test_caso_7_testes_explicitos_do_enunciado_continuam_funcionando(self):
        q = Questao(
            idx=2,
            tipo="modificacao",
            enunciado="Some os dois valores informados.",
            codigo_base="print(0)",
            codigo_aluno_resposta=(
                'a = int(input("A: "))\n'
                'b = int(input("B: "))\n'
                'print(a + b)\n'
            ),
            entradaTestes="A: 7\nB: 8\n",
            saida_esperada="15\n",
        )

        testes = obter_testes(q)

        self.assertEqual(testes[0]["entrada"], "7\n8")
        self.assertEqual(testes[0]["saida"], "15")

    @mock.patch("tests.generator.USAR_LLM", True)
    @mock.patch("tests.generator.chamar_llm_json")
    def test_caso_8_saida_esperada_vem_do_caso_nao_do_codigo_do_aluno(self, mock_llm):
        mock_llm.return_value = {
            "testes": [
                {
                    "valores_entrada": {"opcao": "1", "a": "5", "b": "3"},
                    "saida": "Soma: 8\n",
                    "obs": "soma correta pelo enunciado",
                }
            ]
        }
        q = Questao(
            idx=3,
            tipo="modificacao",
            enunciado="Para a opção 1, o programa deve imprimir 'Soma: 8'.",
            codigo_base="a = int(input())\nb = int(input())\nprint(a + b)",
            codigo_aluno_resposta=(
                'opcao = int(input("Opção: "))\n'
                'a = int(input("A: "))\n'
                'b = int(input("B: "))\n'
                'print("resultado errado")\n'
            ),
        )

        testes = obter_testes(q)

        self.assertEqual(testes[0]["entrada"], "1\n5\n3")
        self.assertEqual(testes[0]["saida"], "Soma: 8")


if __name__ == "__main__":
    unittest.main()
