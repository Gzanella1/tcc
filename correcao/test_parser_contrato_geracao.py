#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 3.1 — Testes do contrato de CORREÇÃO (parser).

Garantias:
    8.  O parser reconhece os novos campos de origem.
    9.  O parser continua aceitando o formato legado.
    10. O código anterior exportado pela geração vira codigo_base.
    11. A resposta futura fica separada da origem.
    12. O arquivo real perguntasGeradas.txt carrega as 5 questões.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.questao import Questao
from parsing.parser import carregar_questoes


BLOCO_NOVO_FORMATO = """
Enunciado original:
Soma de dois números inteiros

Código-base:
----------------------------------------
a = int(input())
b = int(input())
print(a + b)
----------------------------------------

1 - [DESCRITIVA] Descreva o que este código faz.

resposta 1 -
"""

BLOCO_LEGADO = """
1 - [CORRECAO] Qual é o problema com a atualização do valor?

Seu código:
----------------------------------------
valor = int(input("Digite o valor: "))
valor = valor % 100
print(valor)
----------------------------------------

resposta 1 -
O problema é que a atualização precisa preservar o restante correto.
"""


class ParserContratoGeracaoTests(unittest.TestCase):
    def _carregar_texto(self, texto: str) -> list[Questao]:
        with tempfile.TemporaryDirectory() as td:
            caminho = Path(td) / "entrada.txt"
            caminho.write_text(texto, encoding="utf-8")
            return carregar_questoes(caminho)

    # ------------------------------------------------------------------
    # 8 e 10 — Novo formato reconhecido; código anterior vira codigo_base
    # ------------------------------------------------------------------

    def test_novo_formato_captura_codigo_base(self):
        """Provas 8 e 10: codigo_base recebe o código anterior exportado."""
        q = self._carregar_texto(BLOCO_NOVO_FORMATO)[0]

        self.assertEqual(q.tipo, "descritiva")
        self.assertEqual(q.extras.get("enunciado_origem"),
                         "Soma de dois números inteiros")
        self.assertIn("print(a + b)", q.codigo_base)
        self.assertFalse(hasattr(q, "codigo"))

    def test_codigo_base_json_e_preservado(self):
        """Prova 10 (JSON): codigo_base declarado é preservado."""
        conteudo = json.dumps([{
            "id": 1,
            "tipo": "descritiva",
            "enunciado": "a = int(input())\nb = int(input())\nDescreva o fluxo.",
            "enunciado_origem": "Soma",
            "codigo_base": "a = int(input())\nb = int(input())",
        }])

        q = self._carregar_texto(conteudo)[0]

        self.assertIn("a = int(input())", q.codigo_base)
        self.assertEqual(q.extras.get("enunciado_origem"), "Soma")
        self.assertFalse(hasattr(q, "codigo"))

    # ------------------------------------------------------------------
    # 9 — Formato legado intacto
    # ------------------------------------------------------------------

    def test_formato_legado_continua_funcionando(self):
        """Prova 9: 'Seu código:' ainda vira codigo_base como antes."""
        q = self._carregar_texto(BLOCO_LEGADO)[0]

        self.assertEqual(q.tipo, "correcao")
        self.assertIn("valor % 100", q.codigo_base)
        self.assertNotIn("enunciado_origem", q.extras)
        self.assertEqual(
            q.extras.get("resposta_formato"),
            "texto",
        )

    # ------------------------------------------------------------------
    # 11 — Resposta futura separada da origem
    # ------------------------------------------------------------------

    def test_resposta_futura_fica_separada_da_origem(self):
        bloco = BLOCO_NOVO_FORMATO.replace(
            "resposta 1 -\n",
            "resposta 1 -\nEle lê dois números e imprime a soma.\n",
        )
        q = self._carregar_texto(bloco)[0]

        self.assertIn("lê dois números", q.resposta_aluno)
        self.assertIn("print(a + b)", q.codigo_base)
        self.assertNotIn("print(a + b)", q.resposta_aluno)

    # ------------------------------------------------------------------
    # 12 — Arquivo real legado carrega as 5 questões
    # ------------------------------------------------------------------

    def test_arquivo_real_perguntas_geradas_carrega_5_questoes(self):
        real = Path(__file__).parent / ".." / "conteudo" / "perguntasGeradas.txt"
        if not real.exists():
            self.skipTest("conteudo/perguntasGeradas.txt não encontrado")

        questoes = carregar_questoes(real.resolve())

        self.assertEqual(len(questoes), 5)
        tipos = {q.tipo for q in questoes}
        esperados = {"modificacao", "descritiva", "justificativa",
                     "previsao", "correcao"}
        self.assertTrue(esperados.issubset(tipos),
                        f"tipos faltando: {esperados - tipos}")


    # ------------------------------------------------------------------
    # Contrato: pergunta gerada sem resposta = pendente, não erro_entrada
    # ------------------------------------------------------------------

    def test_pergunta_sem_resposta_do_aluno_e_pendente(self):
        """Questão íntegra sem resposta → pendente (não erro_entrada)."""
        from validation import STATUS_PENDENTE, validar_questao

        q = self._carregar_texto(BLOCO_NOVO_FORMATO)[0]
        resultado = validar_questao(q)

        self.assertIsNotNone(resultado)
        self.assertEqual(resultado.status, STATUS_PENDENTE)
        self.assertNotEqual(resultado.status, "erro_entrada")
        self.assertTrue(any("resposta" in d or "codigo" in d
                            for d in resultado.detalhes))

    def test_resposta_real_do_aluno_continua_preenchendo_codigo_aluno(self):
        """Resposta com cerca markdown vira codigo_aluno; validação OK."""
        from validation import validar_questao

        bloco = """
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
        q = self._carregar_texto(bloco)[0]

        # codigo_base continua sendo o código original do exercício
        self.assertIn('input("Nome: ")', q.codigo_base)
        # codigo_aluno veio EXCLUSIVAMENTE da resposta do aluno
        self.assertEqual(
            q.codigo_aluno_resposta,
            'nome = input("Nome: ")\nprint(nome)\nprint(nome)',
        )
        self.assertIsNone(validar_questao(q))

    def test_modificacao_sem_codigo_base_com_resposta_nao_e_erro_de_entrada(self):
        """MODIFICACAO não exige codigo_base quando há código novo e oráculo."""
        from validation import validar_questao

        q = Questao(
            idx=1,
            tipo="modificacao",
            enunciado="Modifique o programa.",
            codigo_base="",
            codigo_aluno_resposta='print("ok")',
            saida_esperada="ok\n",
        )
        resultado = validar_questao(q)

        self.assertIsNone(resultado)


if __name__ == "__main__":
    unittest.main()
