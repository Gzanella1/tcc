#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_evidencia_propagacao.py

Etapa 4.2 — Propagação da fonte da evidência.

Prova que os caminhos ativos de avaliação preenchem Resultado.fonte_evidencia:

    a) codigo.py   → "execucao" (e "ausente" quando não há código)
    b) previsao.py → "execucao" (e "ausente" sem código-base)
    c) texto_llm   → "llm" quando o LLM realmente avaliou
    d) fallback heurístico → "heuristica"
    e) erro_entrada/pendente/inconclusivo → "ausente"
    f) modificacao 70/30 → política documentada: predominante "execucao"
       com componente LLM registrado; só-LLM → "llm"; placeholder → "ausente"
    g) integração via dispatcher: nenhum Resultado ativo com fonte incorreta

O LLM é sempre mockado (unittest.mock); nenhum teste depende do LM Studio.
"""

from __future__ import annotations

import unittest
from unittest import mock

from evaluation.dispatcher import corrigir_questao
from evaluation.evidencia import (
    FONTE_AUSENTE,
    FONTE_EXECUCAO,
    FONTE_HEURISTICA,
    FONTE_LLM,
    FONTES_VALIDAS,
    normalizar_fonte,
)
from evaluation.strategies import codigo as estrategia_codigo
from evaluation.strategies import modificacao as estrategia_modificacao
from evaluation.strategies import previsao as estrategia_previsao
from evaluation.strategies import texto_llm as estrategia_texto_llm
from models.questao import Questao, Resultado
from validation import (
    STATUS_ERRO_ENTRADA,
    STATUS_INCONCLUSIVO,
    STATUS_PENDENTE,
    validar_questao,
)


def _resposta_llm(nota: float = 8.0, status: str = "ok") -> dict:
    return {
        "nota": nota,
        "status": status,
        "feedback": "boa resposta",
        "acertos": ["conceito"],
        "melhorias": [],
    }


class BasePropagacaoTests(unittest.TestCase):
    """Patches defensivos para que NENHUM teste toque em LLM real."""

    def setUp(self) -> None:
        super().setUp()
        self._patchers = [
            mock.patch("tests.generator.USAR_LLM", False),
            mock.patch("tests.generator.chamar_llm_json", return_value=None),
        ]
        for p in self._patchers:
            p.start()
            self.addCleanup(p.stop)

    def _patch_llm(self, alvo: str, retorno):
        p = mock.patch(alvo, return_value=retorno)
        p.start()
        self.addCleanup(p.stop)


class CodigoFonteTests(BasePropagacaoTests):
    # a) codigo.py → fonte "execucao"

    def test_todos_testes_passam_fonte_execucao(self):
        q = Questao(idx=1, tipo="correcao", enunciado="Some dois números.")
        res = estrategia_codigo.avaliar(
            q, "print(1+1)", [{"entrada": "", "saida": "2\n"}]
        )
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.fonte_evidencia, FONTE_EXECUCAO)

    def test_falha_parcial_em_testes_continua_execucao(self):
        q = Questao(idx=1, tipo="correcao", enunciado="Some dois números.")
        res = estrategia_codigo.avaliar(
            q, "print(1+2)", [{"entrada": "", "saida": "2\n"}]
        )
        self.assertEqual(res.fonte_evidencia, FONTE_EXECUCAO)

    def test_sem_testes_execucao_direta_fonte_execucao(self):
        q = Questao(idx=1, tipo="correcao", enunciado="Imprima oi.")
        res = estrategia_codigo.avaliar(q, "print('oi')", [])
        self.assertEqual(res.nota, 10.0)
        self.assertEqual(res.fonte_evidencia, FONTE_EXECUCAO)

    def test_erro_runtime_fonte_execucao(self):
        q = Questao(idx=1, tipo="correcao", enunciado="Divida por zero.")
        res = estrategia_codigo.avaliar(q, "x = 1/0", [])
        self.assertEqual(res.status, "erro")
        self.assertEqual(res.fonte_evidencia, FONTE_EXECUCAO)

    def test_erro_sintaxe_fonte_execucao(self):
        # O compilador Python foi executado sobre o código — evidência real.
        q = Questao(idx=1, tipo="correcao", enunciado="Defina uma função.")
        res = estrategia_codigo.avaliar(q, "def f(:", [])
        self.assertEqual(res.status, "erro")
        self.assertEqual(res.fonte_evidencia, FONTE_EXECUCAO)

    def test_codigo_vazio_nao_inventa_fonte(self):
        q = Questao(idx=1, tipo="correcao", enunciado="Qualquer coisa.")
        res = estrategia_codigo.avaliar(q, "   ", [])
        self.assertEqual(res.fonte_evidencia, FONTE_AUSENTE)


class PrevisaoFonteTests(BasePropagacaoTests):
    # b) previsao.py → fonte "execucao"

    def test_previsao_com_pares_fonte_execucao(self):
        q = Questao(
            idx=2,
            tipo="previsao",
            enunciado="Qual será a saída?",
            codigo_base="n = input()\nprint(int(n) * 2)",
            entrada="3\n",
            resposta_aluno="Entrada: 3\nSaída: 6",
        )
        res = estrategia_previsao.avaliar(q)
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.fonte_evidencia, FONTE_EXECUCAO)

    def test_previsao_modo_legado_fonte_execucao(self):
        q = Questao(
            idx=2,
            tipo="previsao",
            enunciado="Qual será a saída?",
            codigo_base="print(7)",
            entrada="",
            resposta_aluno="7",
        )
        res = estrategia_previsao.avaliar(q)
        self.assertEqual(res.fonte_evidencia, FONTE_EXECUCAO)

    def test_previsao_sem_codigo_base_ausente(self):
        # extrair_codigo usa o próprio enunciado como fallback, logo este
        # caminho só é alcançável sem código E sem enunciado.
        q = Questao(idx=2, tipo="previsao", enunciado="", resposta_aluno="6")
        res = estrategia_previsao.avaliar(q)
        self.assertEqual(res.fonte_evidencia, FONTE_AUSENTE)


class TextoLlmFonteTests(BasePropagacaoTests):
    # c) LLM realmente avaliou → "llm"

    def test_llm_responde_fonte_llm(self):
        self._patch_llm(
            "evaluation.strategies.texto_llm.chamar_llm_json",
            _resposta_llm(),
        )
        q = Questao(
            idx=3,
            tipo="descritiva",
            enunciado="Explique o que é uma variável.",
            rubrica="Menciona armazenamento de valor.",
            resposta_aluno="Variável é um local que armazena um valor.",
        )
        res = estrategia_texto_llm.avaliar(q)
        self.assertEqual(res.fonte_evidencia, FONTE_LLM)

    # d) fallback heurístico → "heuristica"

    def test_llm_falha_sem_conceito_fallback_heuristica(self):
        self._patch_llm("evaluation.strategies.texto_llm.chamar_llm_json", None)
        q = Questao(
            idx=3,
            tipo="justificativa",
            enunciado="Justifique o uso de funções.",
            rubrica="Organização do código.",
            resposta_aluno="Porque sim, ajuda bastante.",
        )
        res = estrategia_texto_llm.avaliar(q)
        self.assertEqual(res.nota, 3.0)
        self.assertEqual(res.fonte_evidencia, FONTE_HEURISTICA)

    def test_llm_falha_com_conceito_fallback_heuristica(self):
        self._patch_llm("evaluation.strategies.texto_llm.chamar_llm_json", None)
        q = Questao(
            idx=3,
            tipo="justificativa",
            enunciado="Por que padronizamos o uso de minusculas e maiusculas?",
            rubrica="Reconhece diferença entre caixas.",
            resposta_aluno="Para diferenciar identificadores, pois Python compara lower e upper.",
        )
        res = estrategia_texto_llm.avaliar(q)
        self.assertEqual(res.nota, 7.5)
        self.assertEqual(res.fonte_evidencia, FONTE_HEURISTICA)

    def test_resposta_vazia_ausente(self):
        q = Questao(idx=3, tipo="descritiva", enunciado="Explique algo.", rubrica="r")
        res = estrategia_texto_llm.avaliar(q)
        self.assertEqual(res.fonte_evidencia, FONTE_AUSENTE)


class ValidationAdministrativoFonteTests(unittest.TestCase):
    # e) resultados administrativos herdam "ausente"

    def test_erro_entrada_ausente(self):
        q = Questao(idx=4, tipo="previsao", enunciado="")
        res = validar_questao(q)
        self.assertIsNotNone(res)
        self.assertEqual(res.status, STATUS_ERRO_ENTRADA)
        self.assertEqual(res.fonte_evidencia, FONTE_AUSENTE)

    def test_pendente_ausente(self):
        q = Questao(
            idx=4,
            tipo="justificativa",
            enunciado="Justifique o uso de listas.",
            rubrica="Menciona ordenação.",
        )
        res = validar_questao(q)
        self.assertIsNotNone(res)
        self.assertEqual(res.status, STATUS_PENDENTE)
        self.assertEqual(res.fonte_evidencia, FONTE_AUSENTE)

    def test_inconclusivo_ausente(self):
        q = Questao(
            idx=4,
            tipo="correcao",
            enunciado="Corrija o código abaixo.",
            codigo_base="x = 1",
            codigo_aluno="x = 2",
        )
        res = validar_questao(q)
        self.assertIsNotNone(res)
        self.assertEqual(res.status, STATUS_INCONCLUSIVO)
        self.assertEqual(res.fonte_evidencia, FONTE_AUSENTE)


class ModificacaoPolitica7030Tests(BasePropagacaoTests):
    # f) modificacao: política documentada no módulo

    def test_com_testes_predominante_execucao_e_componente_llm_registrado(self):
        self._patch_llm(
            "evaluation.strategies.modificacao.chamar_llm_json",
            _resposta_llm(nota=10.0),
        )
        self._patch_llm("evaluation.strategies.modificacao.USAR_LLM", True)
        q = Questao(
            idx=5,
            tipo="modificacao",
            enunciado="Faça o programa imprimir 2.",
            codigo_base="print(1)",
            codigo_aluno="print(2)",
            testes=[{"entrada": "", "saida": "2\n", "obs": ""}],
        )
        with mock.patch(
            "evaluation.strategies.modificacao.USAR_LLM", True
        ):
            res = estrategia_modificacao.avaliar(q)
        self.assertEqual(res.status, "ok")
        # nota = 0.7 * 10 (teste passou) + 0.3 * 10 (LLM mock) = 10
        self.assertEqual(res.nota, 10.0)
        self.assertEqual(res.fonte_evidencia, FONTE_EXECUCAO)
        self.assertTrue(
            any("30% avaliação via LLM" in d for d in res.detalhes),
            msg=f"Componente LLM deve estar nos detalhes: {res.detalhes}",
        )

    def test_sem_testes_nota_depende_so_do_llm_fonte_llm(self):
        self._patch_llm(
            "evaluation.strategies.modificacao.chamar_llm_json",
            _resposta_llm(nota=9.0, status="parcial"),
        )
        q = Questao(
            idx=5,
            tipo="modificacao",
            enunciado="Adicione um comentário ao programa.",
            codigo_base="x = 1",
            codigo_aluno="x = 1  # comentário adicionado",
        )
        with mock.patch(
            "evaluation.strategies.modificacao.USAR_LLM", True
        ):
            res = estrategia_modificacao.avaliar(q)
        self.assertEqual(res.fonte_evidencia, FONTE_LLM)

    def test_llm_desativado_herde_fonte_da_parte_objetiva(self):
        q = Questao(
            idx=5,
            tipo="modificacao",
            enunciado="Faça o programa imprimir 2.",
            codigo_base="print(1)",
            codigo_aluno="print(2)",
            testes=[{"entrada": "", "saida": "2\n", "obs": ""}],
        )
        with mock.patch(
            "evaluation.strategies.modificacao.USAR_LLM", False
        ):
            res = estrategia_modificacao.avaliar(q)
        self.assertEqual(res.fonte_evidencia, FONTE_EXECUCAO)

    def test_placeholder_sem_testes_e_sem_llm_ausente(self):
        q = Questao(
            idx=5,
            tipo="modificacao",
            enunciado="Modifique o programa para ler dois valores.",
            codigo_base="a = input()\nprint(a)",
            codigo_aluno="a = input()\nb = input()\nprint(a, b)",
        )
        with mock.patch(
            "evaluation.strategies.modificacao.USAR_LLM", False
        ):
            res = estrategia_modificacao.avaliar(q)
        self.assertEqual(res.status, "parcial")
        self.assertEqual(res.nota, 0.0)
        self.assertEqual(res.fonte_evidencia, FONTE_AUSENTE)


class DispatcherIntegracaoFonteTests(BasePropagacaoTests):
    # g) nenhum Resultado ativo com fonte incorreta

    def test_dispatcher_previsao_propaga_execucao(self):
        q = Questao(
            idx=6,
            tipo="previsao",
            enunciado="Qual será a saída?",
            codigo_base="n = input()\nprint(n.upper())",
            entrada="abc\n",
            resposta_aluno="ABC",
        )
        self.assertIsNone(validar_questao(q))
        res = corrigir_questao(q)
        self.assertEqual(res.fonte_evidencia, FONTE_EXECUCAO)

    def test_dispatcher_descritiva_via_llm(self):
        self._patch_llm(
            "evaluation.strategies.texto_llm.chamar_llm_json",
            _resposta_llm(),
        )
        q = Questao(
            idx=6,
            tipo="descritiva",
            enunciado="Descreva o que faz um loop for.",
            rubrica="Iteração sobre sequência.",
            resposta_aluno="Um loop for itera sobre os elementos de uma sequência.",
        )
        res = corrigir_questao(q)
        self.assertEqual(res.fonte_evidencia, FONTE_LLM)

    def test_dispatcher_correcao_textual_via_llm(self):
        self._patch_llm(
            "evaluation.strategies.texto_llm.chamar_llm_json",
            _resposta_llm(),
        )
        q = Questao(
            idx=6,
            tipo="correcao",
            enunciado="Qual é o erro no programa?",
            rubrica="Identifica divisão por zero.",
            resposta_aluno="O erro é dividir por zero quando o denominador é 0.",
            extras={"resposta_formato": "texto"},
        )
        res = corrigir_questao(q)
        self.assertEqual(res.fonte_evidencia, FONTE_LLM)

    def test_dispatcher_fallback_heuristico_quando_llm_cai(self):
        self._patch_llm("evaluation.strategies.texto_llm.chamar_llm_json", None)
        q = Questao(
            idx=6,
            tipo="descritiva",
            enunciado="Descreva o que faz um dicionário.",
            rubrica="Estrutura chave-valor.",
            resposta_aluno="É uma estrutura qualquer.",
        )
        res = corrigir_questao(q)
        self.assertEqual(res.fonte_evidencia, FONTE_HEURISTICA)

    def test_dispatcher_pendente_ausente(self):
        q = Questao(
            idx=6,
            tipo="descritiva",
            enunciado="Descreva o que faz uma fila.",
            rubrica="FIFO.",
        )
        res = corrigir_questao(q)
        self.assertEqual(res.status, STATUS_PENDENTE)
        self.assertEqual(res.fonte_evidencia, FONTE_AUSENTE)


class SanidadeFontesValidasTests(BasePropagacaoTests):
    def test_todas_as_fontes_usadas_sao_validas(self):
        amostras = [
            FONTE_EXECUCAO,
            FONTE_LLM,
            FONTE_HEURISTICA,
            FONTE_AUSENTE,
            normalizar_fonte("qualquer"),
        ]
        for fonte in amostras:
            self.assertIn(fonte, FONTES_VALIDAS)


if __name__ == "__main__":
    unittest.main()
