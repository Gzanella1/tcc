#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_evidencia_rastreabilidade.py

Testes da infraestrutura de rastreabilidade da evidência (Etapa 4.1).

Valida:
    - os valores canônicos das fontes definidos em evaluation/evidencia.py;
    - a normalização/validação via normalizar_fonte;
    - o transporte da fonte pelo Resultado sem quebrar o contrato atual
      (construção por palavra-chave e por posição continua funcionando).
"""

from __future__ import annotations

import unittest

from evaluation.evidencia import (
    FONTE_AUSENTE,
    FONTE_EXECUCAO,
    FONTE_HEURISTICA,
    FONTE_LLM,
    FONTES_VALIDAS,
    normalizar_fonte,
)
from models.questao import Resultado


class ConstantesFonteTests(unittest.TestCase):
    def test_valores_canonicos_das_fontes(self):
        self.assertEqual(FONTE_EXECUCAO, "execucao")
        self.assertEqual(FONTE_LLM, "llm")
        self.assertEqual(FONTE_HEURISTICA, "heuristica")
        self.assertEqual(FONTE_AUSENTE, "ausente")

    def test_conjunto_de_fontes_validas_e_fechado(self):
        self.assertEqual(
            FONTES_VALIDAS,
            frozenset({"execucao", "llm", "heuristica", "ausente"}),
        )


class NormalizarFonteTests(unittest.TestCase):
    def test_aceita_todas_as_fontes_validas(self):
        for fonte in ("execucao", "llm", "heuristica", "ausente"):
            self.assertEqual(normalizar_fonte(fonte), fonte)

    def test_normaliza_caixa_e_espacos(self):
        self.assertEqual(normalizar_fonte("  LLM "), "llm")
        self.assertEqual(normalizar_fonte("Execucao"), "execucao")
        self.assertEqual(normalizar_fonte("HEURISTICA"), "heuristica")

    def test_valor_invalido_vira_ausente(self):
        self.assertEqual(normalizar_fonte("chute_do_modelo"), FONTE_AUSENTE)
        self.assertEqual(normalizar_fonte(""), FONTE_AUSENTE)
        self.assertEqual(normalizar_fonte(None), FONTE_AUSENTE)
        self.assertEqual(normalizar_fonte(123), FONTE_AUSENTE)


class ResultadoTransportaFonteTests(unittest.TestCase):
    def test_resultado_sem_fonte_usa_ausente_por_padrao(self):
        res = Resultado(
            idx=1,
            tipo="previsao",
            nota=10.0,
            status="ok",
            feedback="Resposta correta.",
        )
        self.assertEqual(res.fonte_evidencia, FONTE_AUSENTE)

    def test_resultado_aceita_cada_fonte_valida(self):
        for fonte in ("execucao", "llm", "heuristica", "ausente"):
            res = Resultado(
                idx=1,
                tipo="descritiva",
                nota=7.5,
                status="parcial",
                feedback="fb",
                fonte_evidencia=fonte,
            )
            self.assertEqual(res.fonte_evidencia, fonte)

    def test_resultado_normaliza_fonte_invalida_para_ausente(self):
        res = Resultado(
            idx=1,
            tipo="correcao",
            nota=0.0,
            status="erro",
            feedback="fb",
            fonte_evidencia="alucinacao",
        )
        self.assertEqual(res.fonte_evidencia, FONTE_AUSENTE)

    def test_contrato_antigo_por_posicao_continua_funcionando(self):
        # Assinatura original: idx, tipo, nota, status, feedback,
        # detalhes, testes_executados, saida_correta.
        res = Resultado(
            2,
            "justificativa",
            5.0,
            "parcial",
            "fb",
            ["d1"],
            [{"teste": 1}],
            "saida",
        )
        self.assertEqual(res.idx, 2)
        self.assertEqual(res.detalhes, ["d1"])
        self.assertEqual(res.saida_correta, "saida")
        self.assertEqual(res.fonte_evidencia, FONTE_AUSENTE)

    def test_campos_existentes_nao_sao_alterados_pelo_contrato_novo(self):
        res = Resultado(
            idx=3,
            tipo="modificacao",
            nota=8.0,
            status="ok",
            feedback="fb",
            detalhes=["a", "b"],
            testes_executados=[{"teste": 1, "ok": True}],
            saida_correta="x",
            fonte_evidencia=FONTE_EXECUCAO,
        )
        self.assertEqual(res.nota, 8.0)
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.detalhes, ["a", "b"])
        self.assertEqual(res.testes_executados, [{"teste": 1, "ok": True}])
        self.assertEqual(res.saida_correta, "x")


if __name__ == "__main__":
    unittest.main()
