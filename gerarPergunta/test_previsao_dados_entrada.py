#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_previsao_dados_entrada.py

Valida a mudança de contrato da GERAÇÃO de PREVISAO:

    1. dados_entrada (semântico) → entradaTestes (string de stdin) é uma
       transformação DETERMINÍSTICA (services.dados_entrada), sem LLM e
       sem reparse de enunciado.
    2. Serializadores: mapa com 1 dict aninhado (caso dicionário/busca),
       lista linear, dict de escalares; formato desconhecido é rejeitado.
    3. Validação de geração por EXECUÇÃO DE REFERÊNCIA: formato inválido,
       entrada vazia, código que não executa/timeout ou sem saída → a
       pergunta PREVISAO é REJEITADA na geração.
    4. O prompt do builder exige dados_entrada estruturado.
    5. AIClient parseia dados_entrada e, em gerar_pergunta, transforma +
       valida (rejeitando PREVISAO inválida) e carrega entrada_testes.
    6. ReportExporter exporta a seção "entradaTestes:" (JSON string em
       linha única) APENAS para PREVISAO.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from models.exercicio import Exercicio
from models.pergunta import Pergunta
from services.ai_client import AIClient
from services.dados_entrada import (
    FormatoEntradaDesconhecido,
    gerar_entrada_testes,
    serializar_entrada,
)
from services.question_builder import QuestionBuilder
from services.report_exporter import ReportExporter


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

DADOS_ENTRADA_DICIONARIO = {"dicionario": {"a": 5, "b": 3, "c": 5}, "valor_busca": "5"}
ENTRADA_DICIONARIO = "a\n5\nb\n3\nc\n5\n\n5\n"


class SerializacaoTests(unittest.TestCase):
    def test_mapa_com_dict_aninhado_grupo_pares_then_sentinel_then_escalares(self):
        self.assertEqual(
            serializar_entrada(DADOS_ENTRADA_DICIONARIO),
            ENTRADA_DICIONARIO,
        )

    def test_lista_linear_uma_linha_por_elemento(self):
        self.assertEqual(
            serializar_entrada(["João", "14:00"]),
            "João\n14:00\n",
        )

    def test_lista_com_aninhamento(self):
        self.assertEqual(
            serializar_entrada([["a", 1], "fim"]),
            "a\n1\nfim\n",
        )

    def test_dict_de_escalares_uma_linha_por_valor(self):
        self.assertEqual(
            serializar_entrada({"numeros": [2, 3, 4]}),
            "2\n3\n4\n",
        )

    def test_formato_desconhecido_e_rejeitado(self):
        with self.assertRaises(FormatoEntradaDesconhecido):
            serializar_entrada("a\n5\n")      # string não é dado semântico
        with self.assertRaises(FormatoEntradaDesconhecido):
            serializar_entrada(None)          # ausente
        with self.assertRaises(FormatoEntradaDesconhecido):
            serializar_entrada(42)

    def test_dict_vazio_nao_produz_entrada(self):
        self.assertEqual(serializar_entrada({}), "")


class ValidacaoExecucaoTests(unittest.TestCase):
    def test_caso_dicionario_valida_e_produz_saida_esperada(self):
        ok, motivo, entrada, saida = gerar_entrada_testes(
            DADOS_ENTRADA_DICIONARIO, CODIGO_BASE_DICIONARIO
        )

        self.assertTrue(ok, motivo)
        self.assertEqual(entrada, ENTRADA_DICIONARIO)
        self.assertIn("['a', 'c']", saida)

    def test_vetor_lista_valida(self):
        codigo = "for _ in range(3):\n    print(int(input()))"
        ok, motivo, entrada, _saida = gerar_entrada_testes([2, 3, 4], codigo)

        self.assertTrue(ok, motivo)
        self.assertEqual(entrada, "2\n3\n4\n")

    def test_formato_invalido_e_rejeitado(self):
        ok, motivo, entrada, _saida = gerar_entrada_testes(
            "a\n5\n", CODIGO_BASE_DICIONARIO
        )

        self.assertFalse(ok)
        self.assertEqual(entrada, "")
        self.assertIn("não suportado", motivo)

    def test_entrada_vazia_e_rejeitada(self):
        ok, motivo, entrada, _saida = gerar_entrada_testes({}, CODIGO_BASE_DICIONARIO)

        self.assertFalse(ok)
        self.assertIn("não produziu entrada", motivo)

    def test_codigo_que_nao_executa_e_rejeitado(self):
        ok, motivo, _entrada, _saida = gerar_entrada_testes(
            ["1"], "print("      # sintaxe inválida
        )

        self.assertFalse(ok)
        self.assertIn("não executou", motivo)

    def test_timeout_e_rejeitado(self):
        codigo = (
            "import time\n"
            "time.sleep(30)\n"
            "print('nunca chega aqui')\n"
        )
        ok, motivo, _entrada, _saida = gerar_entrada_testes(
            ["1"], codigo, timeout=1
        )

        self.assertFalse(ok)
        self.assertIn("timeout", motivo)


class QuestionBuilderTests(unittest.TestCase):
    def test_prompt_previsao_exige_dados_entrada_estruturado(self):
        ex = Exercicio(numero=1, titulo="Soma", codigo_base="print(1 + 2)")
        prompt = QuestionBuilder().construir(ex, "previsao")

        self.assertIn("CAMPO 'dados_entrada'", prompt)
        self.assertIn('"dados_entrada": <objeto JSON com os dados>', prompt)
        self.assertIn('"tipo": "previsao"', prompt)
        # regras obrigatórias continuam presentes
        self.assertIn("Não forneça a resposta", prompt)


class AIClientTransformTests(unittest.TestCase):
    def _exercicio(self) -> Exercicio:
        return Exercicio(
            numero=1,
            titulo="Busca em dicionário",
            codigo_base=CODIGO_BASE_DICIONARIO,
        )

    def test_parsear_resposta_preserva_dados_entrada(self):
        texto = (
            '[{"tipo": "previsao", "pergunta": "Preveja a saída.", '
            '"dados_entrada": {"dicionario": {"a": 5}, "valor_busca": "5"}}]'
        )
        perguntas = AIClient()._parsear_resposta(texto, exercicio_numero=1)

        self.assertEqual(len(perguntas), 1)
        self.assertEqual(
            perguntas[0].dados_entrada,
            {"dicionario": {"a": 5}, "valor_busca": "5"},
        )

    def test_gerar_pergunta_previsao_transform_e_valida(self):
        falsa = Pergunta(
            tipo="previsao",
            pergunta="Preveja a saída.",
            exercicio_numero=1,
            dados_entrada=DADOS_ENTRADA_DICIONARIO,
        )
        with patch.object(AIClient, "_chamar_api_cached", return_value=[falsa]):
            resultado = AIClient().gerar_pergunta(self._exercicio(), "previsao")

        self.assertEqual(len(resultado), 1)
        p = resultado[0]
        self.assertEqual(p.entrada_testes, ENTRADA_DICIONARIO)
        self.assertEqual(p.dados_entrada, DADOS_ENTRADA_DICIONARIO)
        self.assertEqual(p.enunciado_origem, "Busca em dicionário")

    def test_gerar_pergunta_rejeita_previsao_invalida(self):
        falsa = Pergunta(
            tipo="previsao",
            pergunta="Preveja a saída.",
            exercicio_numero=1,
            dados_entrada="a\n5\n",     # dado como STRING → formato inválido
        )
        with patch.object(AIClient, "_chamar_api_cached", return_value=[falsa]):
            resultado = AIClient().gerar_pergunta(self._exercicio(), "previsao")

        self.assertEqual(resultado, [])

    def test_gerar_pergunta_nao_previsao_nao_altera_comportamento(self):
        falsa = Pergunta(tipo="descritiva", pergunta="Descreva.", exercicio_numero=1)
        with patch.object(AIClient, "_chamar_api_cached", return_value=[falsa]):
            resultado = AIClient().gerar_pergunta(self._exercicio(), "descritiva")

        self.assertEqual(len(resultado), 1)
        self.assertEqual(resultado[0].entrada_testes, "")
        self.assertIsNone(resultado[0].dados_entrada)


class ReportExporterTests(unittest.TestCase):
    def test_exporta_secao_entrada_testes_apenas_para_previsao(self):
        ex = Exercicio(numero=1, titulo="Busca", codigo_base=CODIGO_BASE_DICIONARIO)

        p_pre = Pergunta(
            tipo="previsao",
            pergunta="Preveja a saída.",
            exercicio_numero=1,
            entrada_testes=ENTRADA_DICIONARIO,
        )
        p_desc = Pergunta(
            tipo="descritiva",
            pergunta="Descreva.",
            exercicio_numero=1,
            entrada_testes="",           # outros tipos não têm a seção
        )

        texto = ReportExporter(caminho_saida_tmp())._formatar([(ex, p_pre)])
        self.assertIn(f"entradaTestes: {json.dumps(ENTRADA_DICIONARIO)}", texto)
        # linha única, parseável como JSON string
        for linha in texto.splitlines():
            if linha.startswith("entradaTestes: "):
                valor = json.loads(linha.split("entradaTestes: ", 1)[1])
                self.assertEqual(valor, ENTRADA_DICIONARIO)

        texto_desc = ReportExporter(caminho_saida_tmp())._formatar([(ex, p_desc)])
        self.assertNotIn("entradaTestes:", texto_desc)

    def test_previsao_sem_entrada_testes_nao_emite_secao(self):
        ex = Exercicio(numero=1, titulo="Busca", codigo_base=CODIGO_BASE_DICIONARIO)
        p = Pergunta(tipo="previsao", pergunta="Preveja a saída.", exercicio_numero=1)
        texto = ReportExporter(caminho_saida_tmp())._formatar([(ex, p)])
        self.assertNotIn("entradaTestes:", texto)


def caminho_saida_tmp():
    import tempfile
    from pathlib import Path
    return str(Path(tempfile.gettempdir()) / "tcc_previsao_dados_entrada_teste_saida.txt")


if __name__ == "__main__":
    unittest.main()