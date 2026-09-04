#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 3.1 — Testes do contrato de GERAÇÃO.

Garantias:
    1. O loader captura o enunciado original ("N - Título").
    2. O loader separa o código anterior do aluno dos enunciados.
    3. O Exercicio usa codigo_base, sem aliases de código.
    4. O builder recebe explicitamente enunciado original + código.
    5. A Pergunta preserva enunciado_origem.
    6. O exportador emite Código-base para a correção.
    7. O exportador NUNCA rotula o contexto como "Seu código:".
    12. O arquivo real conhecimento.txt carrega as 5 questões.
"""

from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.exercicio import Exercicio
from models.pergunta import Pergunta
from services.knowledge_loader import KnowledgeLoader
from services.question_builder import QuestionBuilder
from services.report_exporter import ReportExporter
from services.ai_client import AIClient


FIXTURE_CONHECIMENTO = """\
Questões:

1 - Soma de dois números inteiros
2 - Média de três notas

resposta 1 -
a = int(input())
b = int(input())
print(a + b)

resposta 2-
n1 = float(input())
n2 = float(input())
n3 = float(input())
print((n1 + n2 + n3) / 3)

1 - linha dentro de código não pode virar título
"""


class ContratoGeracaoTests(unittest.TestCase):
    """Prova os 8 requisitos do lado da geração."""

    # ------------------------------------------------------------------
    # 1 e 2 — Loader
    # ------------------------------------------------------------------

    def test_loader_captura_enunciado_original(self):
        """Prova 1: 'N - Enunciado' vira titulo/enunciado_original."""
        with tempfile_dir() as caminho:
            caminho.write_text(FIXTURE_CONHECIMENTO, encoding="utf-8")
            exercicios = KnowledgeLoader(str(caminho)).carregar()

        self.assertEqual(len(exercicios), 2)
        self.assertEqual(exercicios[0].enunciado_original,
                         "Soma de dois números inteiros")
        self.assertEqual(exercicios[0].titulo,
                         "Soma de dois números inteiros")
        self.assertEqual(exercicios[1].enunciado_original,
                         "Média de três notas")

    def test_loader_separa_codigo_anterior_do_enunciado(self):
        """Prova 2: blocos 'resposta N' viram codigo_base."""
        with tempfile_dir() as caminho:
            caminho.write_text(FIXTURE_CONHECIMENTO, encoding="utf-8")
            exercicios = KnowledgeLoader(str(caminho)).carregar()

        self.assertIn("print(a + b)", exercicios[0].codigo_base)
        self.assertIn("(n1 + n2 + n3)", exercicios[1].codigo_base)
        # linha "1 - ..." dentro de bloco resposta NUNCA vira título
        self.assertNotIn("linha dentro de código",
                         exercicios[0].enunciado_original)
        self.assertNotIn("linha dentro de código",
                         exercicios[1].enunciado_original)

    # ------------------------------------------------------------------
    # 3 — Contrato sem aliases de código
    # ------------------------------------------------------------------

    def test_exercicio_usa_codigo_base_sem_aliases_de_codigo(self):
        ex = Exercicio(numero=1, titulo="T", codigo_base="print(1)")

        self.assertEqual(ex.codigo_base, "print(1)")
        self.assertFalse(hasattr(ex, "codigo"))
        self.assertFalse(hasattr(ex, "codigo_aluno_anterior"))

    def test_alias_titulo_enunciado_original_bidirecional(self):
        ex = Exercicio(numero=1, titulo="Soma")
        self.assertEqual(ex.enunciado_original, "Soma")
        ex.enunciado_original = "Somar"
        self.assertEqual(ex.titulo, "Somar")

    # ------------------------------------------------------------------
    # 4 — Builder recebe dados com papéis explícitos
    # ------------------------------------------------------------------

    def test_builder_prompt_rotula_enunciado_e_codigo(self):
        """Prova 4: prompt contém rótulos explícitos e conteúdos corretos."""
        ex = Exercicio(
            numero=7,
            titulo="Soma",
            codigo_base="a = 1\nb = 2\nprint(a + b)",
        )
        prompt = QuestionBuilder().construir(ex, "justificativa")

        self.assertIn("Enunciado original:\nSoma", prompt)
        self.assertIn(
            "Código anterior do aluno (contexto/apoio; não é gabarito):\n"
            "a = 1\nb = 2\nprint(a + b)",
            prompt,
        )
        # rótulo antigo ambíguo não deve mais aparecer
        self.assertNotIn("Código do aluno:\n", prompt)

    # ------------------------------------------------------------------
    # 5 — Transporte da origem textual na Pergunta
    # ------------------------------------------------------------------

    def test_pergunta_preserva_origem_textual_sem_duplicar_codigo_base(self):
        """Prova 5: Pergunta carrega enunciado_origem e não duplica codigo_base."""
        ex = Exercicio(
            numero=3,
            titulo="Maior de três",
            codigo_base="maior = None\nfor x in [1, 2]:\n    if maior is None or x > maior:\n        maior = x",
        )
        falsa = Pergunta(tipo="descritiva", pergunta="O que o código faz?",
                         exercicio_numero=3)

        with patch.object(AIClient, "_chamar_api_cached", return_value=[falsa]):
            cliente = AIClient()
            resultado = cliente.gerar_pergunta(ex, "descritiva")

        self.assertEqual(len(resultado), 1)
        p = resultado[0]
        self.assertEqual(p.enunciado_origem, "Maior de três")
        self.assertFalse(hasattr(p, "codigo_aluno_origem"))
        # sem evidência inventada nesta fase
        self.assertIsNone(p.conceito_avaliado)
        self.assertIsNone(p.evidencia_codigo)

    # ------------------------------------------------------------------
    # 7 — Exportador nunca rotula origem como "Seu código:"
    # ------------------------------------------------------------------

    def test_exporter_usa_rotulo_de_origem_e_mantem_resposta_futura(self):
        """Prova 7: Código-base presente; 'Seu código:' ausente."""
        ex = Exercicio(numero=1, titulo="Soma", codigo_base="print(a + b)")
        p = Pergunta(tipo="previsao", pergunta="Qual a saída para 2 e 3?",
                     exercicio_numero=1,
                     enunciado_origem="Soma")
        texto = ReportExporter(caminho_saida_tmp())._formatar([(ex, p)])

        self.assertIn("Código-base:", texto)
        self.assertIn("Enunciado original:", texto)
        self.assertIn("print(a + b)", texto)
        self.assertNotIn("Seu código:", texto)
        # cabeçalho compatível com split_exercicios da correção
        self.assertIn("Exercício gerado com base na sua resposta da questão 1", texto)
        # seção de resposta futura presente
        self.assertIn("resposta 1 -", texto)

    # ------------------------------------------------------------------
    # 12 — Arquivo real
    # ------------------------------------------------------------------

    def test_arquivo_real_conhecimento_carrega_5_exercicios(self):
        real = Path(__file__).parent / ".." / "conteudo" / "conhecimento.txt"
        if not real.exists():
            self.skipTest("conteudo/conhecimento.txt não encontrado")

        exercicios = KnowledgeLoader(str(real.resolve())).carregar()

        self.assertEqual(len(exercicios), 5)
        for ex in exercicios:
            self.assertTrue(ex.enunciado_original,
                            f"Exercício {ex.numero} sem enunciado original")
            self.assertNotEqual(ex.enunciado_original, f"Questão {ex.numero}")
            self.assertTrue(ex.codigo_base,
                            f"Exercício {ex.numero} sem código anterior")


# ----------------------------------------------------------------------
# Utilitários
# ----------------------------------------------------------------------

@contextlib.contextmanager
def tempfile_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td) / "conhecimento.txt"


def caminho_saida_tmp():
    return str(Path(tempfile.gettempdir()) / "tcc_fase31_teste_saida.txt")


if __name__ == "__main__":
    unittest.main()
