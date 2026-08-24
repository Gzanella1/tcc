#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.questao import Questao
from parsing.parser import carregar_questoes
from validation import (
    STATUS_ERRO_ENTRADA,
    STATUS_INCONCLUSIVO,
    resposta_correcao_eh_textual,
    validar_questao,
)


class ParserValidationTests(unittest.TestCase):
    def _carregar_texto(self, texto: str) -> list[Questao]:
        with tempfile.TemporaryDirectory() as td:
            caminho = Path(td) / "entrada.txt"
            caminho.write_text(texto, encoding="utf-8")
            return carregar_questoes(caminho)

    def test_json_valido_com_schema_canonico(self):
        conteudo = json.dumps({
            "questoes": [{
                "id": 10,
                "tipo": "previsao",
                "enunciado": "Qual sera a saida?",
                "codigo_base": "nome = input()\nprint(nome)",
                "entrada": "Ana\n",
                "saida_esperada": "Ana\n",
                "resposta_aluno": "Ana",
            }]
        })

        q = self._carregar_texto(conteudo)[0]

        self.assertEqual(q.idx, 10)
        self.assertEqual(q.codigo_base, "nome = input()\nprint(nome)")
        self.assertEqual(q.codigo, q.codigo_base)
        self.assertEqual(q.saida_esperada, "Ana")
        self.assertEqual(q.saida, q.saida_esperada)
        self.assertIsNone(validar_questao(q))

    def test_json_invalido_gera_questao_com_erro_de_entrada(self):
        q = self._carregar_texto("{ isto nao e json")[0]

        resultado = validar_questao(q)

        self.assertIsNotNone(resultado)
        self.assertEqual(resultado.status, STATUS_ERRO_ENTRADA)

    def test_questao_textual_de_correcao_nao_vira_codigo(self):
        texto = """
1 - [CORRECAO] Qual é o problema com a atualização do valor?

Seu código:
----------------------------------------
valor = int(input("Digite o valor: "))
valor = valor % 100
print(valor)
----------------------------------------

resposta 1 -
O problema é que a atualização do valor precisa preservar o restante correto.
"""

        q = self._carregar_texto(texto)[0]

        self.assertEqual(q.tipo, "correcao")
        self.assertTrue(q.codigo_base)
        self.assertEqual(q.codigo_aluno, "")
        self.assertEqual(q.extras.get("resposta_formato"), "texto")
        self.assertTrue(resposta_correcao_eh_textual(q))
        self.assertIsNone(validar_questao(q))

    def test_previsao_extrai_vetor_do_enunciado_como_entrada(self):
        texto = """
1 - [PREVISAO] Dado o vetor [2, 3, 4, 5], qual será a saída esperada?

Seu código:
----------------------------------------
for i in range(4):
    numero = int(input())
    print(numero)
----------------------------------------

resposta 1 -
2
3
4
5
"""

        q = self._carregar_texto(texto)[0]

        self.assertEqual(q.entrada, "2\n3\n4\n5\n")
        self.assertIsNone(validar_questao(q))

    def test_modificacao_separa_codigo_base_e_codigo_aluno(self):
        texto = """
1 - [MODIFICACAO] Modifique o programa para repetir o nome digitado.

Seu código:
----------------------------------------
nome = input("Nome: ")
print(nome)
----------------------------------------

resposta 1 -
```python
nome = input("Nome: ")
print(nome)
print(nome)
```
"""

        q = self._carregar_texto(texto)[0]

        self.assertIn("print(nome)", q.codigo_base)
        self.assertEqual(q.codigo_aluno, 'nome = input("Nome: ")\nprint(nome)\nprint(nome)')
        self.assertEqual(q.extras.get("resposta_formato"), "codigo")
        self.assertIsNone(validar_questao(q))

    def test_codigo_base_ausente_e_erro_de_entrada(self):
        q = Questao(
            idx=1,
            tipo="previsao",
            enunciado="Qual sera a saida?",
            entrada="1\n",
            resposta_aluno="1",
        )

        resultado = validar_questao(q)

        self.assertIsNotNone(resultado)
        self.assertEqual(resultado.status, STATUS_ERRO_ENTRADA)
        self.assertTrue(any("codigo_base" in d for d in resultado.detalhes))

    def test_entrada_ausente_e_erro_de_entrada(self):
        q = Questao(
            idx=1,
            tipo="previsao",
            enunciado="Qual sera a saida?",
            codigo_base="print(input())",
            resposta_aluno="1",
        )

        resultado = validar_questao(q)

        self.assertIsNotNone(resultado)
        self.assertEqual(resultado.status, STATUS_ERRO_ENTRADA)
        self.assertTrue(any("entrada" in d for d in resultado.detalhes))

    def test_testes_ausentes_em_codigo_sem_input_sao_inconclusivos(self):
        q = Questao(
            idx=1,
            tipo="modificacao",
            enunciado="Modifique o programa para imprimir B.",
            codigo_base='print("A")',
            codigo_aluno='print("B")',
        )

        resultado = validar_questao(q)

        self.assertIsNotNone(resultado)
        self.assertEqual(resultado.status, STATUS_INCONCLUSIVO)

    def test_cercas_markdown_extraem_codigo_do_aluno(self):
        conteudo = json.dumps([{
            "tipo": "modificacao",
            "enunciado": "Modifique para ler e imprimir o nome duas vezes.",
            "codigo_base": "nome = input()\nprint(nome)",
            "resposta_aluno": "```python\nnome = input()\nprint(nome)\nprint(nome)\n```",
        }])

        q = self._carregar_texto(conteudo)[0]

        self.assertEqual(q.codigo_aluno, "nome = input()\nprint(nome)\nprint(nome)")
        self.assertEqual(q.extras.get("resposta_formato"), "codigo")
        self.assertIsNone(validar_questao(q))

    def test_json_explicito_preserva_codigo_base_codigo_aluno_e_resposta(self):
        conteudo = json.dumps([{
            "tipo": "modificacao",
            "enunciado": "Modifique o programa.",
            "codigo_base": "print('base')",
            "codigo_aluno": "print('aluno')",
            "resposta_aluno": "Resposta em texto livre.",
        }])

        q = self._carregar_texto(conteudo)[0]

        self.assertEqual(q.codigo_base, "print('base')")
        self.assertEqual(q.codigo_aluno, "print('aluno')")
        self.assertEqual(q.resposta_aluno, "Resposta em texto livre.")
        self.assertEqual(q.extras.get("resposta_formato"), "codigo")

    def test_json_legado_com_codigo_preserva_compatibilidade(self):
        conteudo = json.dumps([{
            "tipo": "descritiva",
            "enunciado": "Descreva o codigo.",
            "codigo": "print('base')",
            "resposta_aluno": "Ele imprime algo.",
        }])

        q = self._carregar_texto(conteudo)[0]

        self.assertEqual(q.codigo_base, "print('base')")
        self.assertEqual(q.codigo, "print('base')")
        self.assertEqual(q.codigo_aluno, "")
        self.assertEqual(q.resposta_aluno, "Ele imprime algo.")
        self.assertEqual(q.extras.get("resposta_formato"), "texto")

    def test_formato_atual_de_perguntas_geradas(self):
        texto = """
============================================================
Exercício gerado com base na sua resposta da questão 1: Questão 1
============================================================

1 - [DESCRITIVA] Descreva o que o código faz.

Seu código:
----------------------------------------
print("ola")
----------------------------------------

resposta 1 -
Ele imprime uma mensagem.

============================================================
Exercício gerado com base na sua resposta da questão 2: Questão 2
============================================================

2 - [PREVISAO] Dado o vetor [1, 2], qual será a saída?

Seu código:
----------------------------------------
print(int(input()))
print(int(input()))
----------------------------------------

resposta 2 -
1
2
"""

        questoes = self._carregar_texto(texto)

        self.assertEqual(len(questoes), 2)
        self.assertEqual(questoes[0].tipo, "descritiva")
        self.assertEqual(questoes[0].codigo_base, 'print("ola")')
        self.assertTrue(questoes[0].rubrica)
        self.assertEqual(questoes[1].entrada, "1\n2\n")
        self.assertIsNone(validar_questao(questoes[0]))
        self.assertIsNone(validar_questao(questoes[1]))


if __name__ == "__main__":
    unittest.main()
