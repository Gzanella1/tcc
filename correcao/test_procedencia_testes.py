#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_procedencia_testes.py

Etapa 4.4 — Procedência da régua de testes.

Valida que os casos de teste usados como régua de avaliação carregam a
chave INTERNA "_origem" ("enunciado" | "llm") ao longo do fluxo de
geração/validação/deduplicação, e que a evidência de execução registra
"testes_por_origem" sem alterar nota, status, pesos ou interfaces públicas:

    1. teste explícito            → "_origem": "enunciado"
    2. teste gerado por LLM       → "_origem": "llm"
    3. validar_testes preserva    → "_origem"
    4. deduplicar_testes preserva → "_origem"
    5. colisão na dedup           → sobrevivente mantém a própria origem
    6. obter_testes()             → mesma assinatura/comportamento público
    7. testes sem "_origem"       → seguem funcionando (tratados como
                                    "enunciado" na contagem)
    8. codigo.py                  → "testes_por_origem": {"enunciado": X,
                                    "llm": Y}
    9. X + Y                      → igual ao total efetivamente utilizado
    10. modificacao.py            → evidência de execução com peso 0.7
    11. vazamento                 → nenhum "_origem" em testes_executados
                                    nem no relatório público
    12. serialização              → evidências JSON-serializáveis

Nenhum teste depende de LM Studio ou LLM real (unittest.mock).
"""

from __future__ import annotations

import inspect
import json
import unittest
from unittest import mock

from evaluation.strategies import codigo as estrategia_codigo
from evaluation.strategies import modificacao as estrategia_modificacao
from models.questao import Questao
from report.formatter import formatar_resultado
from tests.generator import (
    ORIGEM_ENUNCIADO,
    ORIGEM_LLM,
    deduplicar_testes,
    obter_testes,
    obter_testes_explicitos,
    validar_testes,
)


def _resposta_llm_requisitos(nota: float = 10.0, status: str = "ok") -> dict:
    return {
        "nota": nota,
        "status": status,
        "cumpre_requisitos": True,
        "requisitos_identificados": ["atende"],
        "requisitos_atendidos": ["atende"],
        "faltantes": [],
        "feedback": "ok",
    }


class BaseProcedenciaTests(unittest.TestCase):
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


class MarcacaoOrigemTests(BaseProcedenciaTests):
    # 1) explícito → "enunciado"

    def test_caso_montado_do_enunciado_recebe_origem_enunciado(self):
        q = Questao(
            idx=1,
            tipo="correcao",
            enunciado="Imprima o dobro.",
            codigo_base="n = int(input())\nprint(n * 2)",
            entradaTestes="3\n",
            saida_esperada="6\n",
        )
        testes = obter_testes_explicitos(q)
        self.assertEqual(len(testes), 1)
        self.assertEqual(testes[0]["_origem"], ORIGEM_ENUNCIADO)

    def test_testes_da_questao_recebem_origem_enunciado_sem_mutar_original(self):
        original = {"entrada": "", "saida": "2\n", "obs": ""}
        q = Questao(
            idx=1,
            tipo="correcao",
            enunciado="Imprima 2.",
            codigo_base="print(2)",
            testes=[original],
        )
        testes = obter_testes_explicitos(q)
        self.assertEqual(len(testes), 1)
        self.assertEqual(testes[0]["_origem"], ORIGEM_ENUNCIADO)
        # Retrocompatibilidade: o dicionário original da Questao não é mutado.
        self.assertNotIn("_origem", original)

    # 2) gerado pelo LLM → "llm"

    def test_teste_gerado_por_llm_recebe_origem_llm(self):
        self._patch_llm("tests.generator.USAR_LLM", True)
        self._patch_llm(
            "tests.generator.chamar_llm_json",
            {"testes": [{"entrada": "oi\n", "saida": "oi\n", "obs": ""}]},
        )
        q = Questao(
            idx=2,
            tipo="correcao",
            enunciado="Repita a palavra.",
            codigo_base="p = input()\nprint(p)",
        )
        testes = obter_testes(q)
        self.assertEqual(len(testes), 1)
        self.assertEqual(testes[0]["_origem"], ORIGEM_LLM)


class PreservacaoOrigemTests(BaseProcedenciaTests):
    # 3) validar_testes preserva

    def test_validar_testes_preserva_origem(self):
        rotulados = [
            {"entrada": "a\n", "saida": "a\n", "obs": "", "_origem": ORIGEM_ENUNCIADO},
            {"entrada": "b\n", "saida": "b\n", "obs": "", "_origem": ORIGEM_LLM},
        ]
        validos = validar_testes(rotulados, requer_input=True)
        self.assertEqual(validos[0]["_origem"], ORIGEM_ENUNCIADO)
        self.assertEqual(validos[1]["_origem"], ORIGEM_LLM)

    # 4) deduplicar_testes preserva

    def test_deduplicar_testes_preserva_origem(self):
        rotulados = [
            {"entrada": "a\n", "saida": "a\n", "obs": "", "_origem": ORIGEM_ENUNCIADO},
            {"entrada": "b\n", "saida": "b\n", "obs": "", "_origem": ORIGEM_LLM},
        ]
        unicos = deduplicar_testes(rotulados)
        self.assertEqual([t["_origem"] for t in unicos], [ORIGEM_ENUNCIADO, ORIGEM_LLM])

    # 5) colisão: origem acompanha o sobrevivente (primeiro permanece)

    def test_colisao_dedup_sobrevivente_mantem_propria_origem(self):
        explicito_primeiro = [
            {"entrada": "a\n", "saida": "s1\n", "obs": "", "_origem": ORIGEM_ENUNCIADO},
            {"entrada": "a ", "saida": "s2\n", "obs": "", "_origem": ORIGEM_LLM},
        ]
        unicos = deduplicar_testes(explicito_primeiro)
        self.assertEqual(len(unicos), 1)
        self.assertEqual(unicos[0]["entrada"], "a")
        self.assertEqual(unicos[0]["_origem"], ORIGEM_ENUNCIADO)

        llm_primeiro = [
            {"entrada": "a ", "saida": "s2\n", "obs": "", "_origem": ORIGEM_LLM},
            {"entrada": "a\n", "saida": "s1\n", "obs": "", "_origem": ORIGEM_ENUNCIADO},
        ]
        unicos = deduplicar_testes(llm_primeiro)
        self.assertEqual(len(unicos), 1)
        self.assertEqual(unicos[0]["_origem"], ORIGEM_LLM)


class CompatibilidadePublicaTests(BaseProcedenciaTests):
    # 6) assinatura pública de obter_testes()

    def test_obter_testes_mantem_assinatura_publica(self):
        sig = inspect.signature(obter_testes)
        self.assertEqual(list(sig.parameters), ["q"])

    def test_obter_testes_retorna_lista_simples_de_dicionarios(self):
        res = obter_testes(Questao(idx=3, tipo="correcao", enunciado="x"))
        self.assertIsInstance(res, list)
        self.assertEqual(res, [])

    # 7) testes sem "_origem" continuam funcionando

    def test_validar_e_deduplicar_sem_origem_nao_inventam_chave(self):
        brutos = [{"entrada": "a\n", "saida": "a\n", "obs": ""}]
        validos = validar_testes(brutos, requer_input=True)
        self.assertNotIn("_origem", validos[0])
        unicos = deduplicar_testes(brutos)
        self.assertNotIn("_origem", unicos[0])

    def test_avaliar_com_testes_sem_origem_funciona(self):
        q = Questao(idx=4, tipo="correcao", enunciado="Some.")
        res = estrategia_codigo.avaliar(
            q, "print(1+1)", [{"entrada": "", "saida": "2\n"}]
        )
        self.assertEqual(res.status, "ok")
        self.assertEqual(
            res.evidencias[0]["dados"]["testes_por_origem"],
            {"enunciado": 1, "llm": 0},
        )


class EvidenciaProcedenciaTests(BaseProcedenciaTests):
    """8, 9 e 11 — contagem por origem na evidência de execução."""

    def _questao_mista(self) -> Questao:
        return Questao(
            idx=7,
            tipo="correcao",
            enunciado="Dobre o número lido.",
            codigo_base="x = input()\nprint(int(x) * 2)",
            entradaTestes="3\n",
            saida_esperada="6\n",
        )

    def _testes_mistos(self, q: Questao):
        """1 caso explícito + 2 casos LLM (mockados) via fluxo real."""
        self._patch_llm(
            "tests.generator.chamar_llm_json",
            {
                "testes": [
                    {"entrada": "4\n", "saida": "8\n", "obs": ""},
                    {"entrada": "5\n", "saida": "10\n", "obs": ""},
                ]
            },
        )
        self._patch_llm("tests.generator.USAR_LLM", True)
        return obter_testes(q)

    def test_contagem_por_origem_na_evidencia(self):
        q = self._questao_mista()
        testes = self._testes_mistos(q)
        origens = [t["_origem"] for t in testes]
        self.assertEqual(origens.count(ORIGEM_ENUNCIADO), 1)
        self.assertEqual(origens.count(ORIGEM_LLM), 2)

        res = estrategia_codigo.avaliar(
            q, "x = input()\nprint(int(x) * 2)", testes
        )
        ev = res.evidencias[0]
        self.assertEqual(ev["tipo"], "execucao")
        self.assertEqual(ev["dados"]["modo"], "com_testes")
        # 8) ambos os campos existem sempre no modo com_testes
        self.assertEqual(
            ev["dados"]["testes_por_origem"],
            {"enunciado": 1, "llm": 2},
        )
        # 9) X + Y corresponde ao total efetivamente utilizado
        total = ev["dados"]["testes_total"]
        soma = sum(ev["dados"]["testes_por_origem"].values())
        self.assertEqual(soma, total)
        self.assertEqual(total, len(res.testes_executados))

    # 11) "_origem" não vaza para testes_executados nem para o relatório

    def test_origem_interna_nao_vaza(self):
        q = self._questao_mista()
        testes = self._testes_mistos(q)
        res = estrategia_codigo.avaliar(
            q, "x = input()\nprint(int(x) * 2)", testes
        )
        for execucao in res.testes_executados:
            self.assertNotIn("_origem", execucao)
        relatorio = formatar_resultado(res, q)
        # A chave interna nunca aparece como chave renderizada; o sufixo de
        # "testes_por_origem" (público) é permitido.
        self.assertNotIn("'_origem'", relatorio)

    # 12) evidências continuam JSON-serializáveis

    def test_evidencias_serializaveis(self):
        q = self._questao_mista()
        testes = self._testes_mistos(q)
        res = estrategia_codigo.avaliar(
            q, "x = input()\nprint(int(x) * 2)", testes
        )
        texto = json.dumps(res.evidencias, ensure_ascii=False)
        self.assertIn("testes_por_origem", texto)


class ModificacaoHerancaProcedenciaTests(BaseProcedenciaTests):
    # 10) evidência de execução com peso 0.7 carrega testes_por_origem

    def test_modificacao_preserva_peso_e_procedencia(self):
        self._patch_llm("evaluation.strategies.modificacao.USAR_LLM", True)
        self._patch_llm(
            "evaluation.strategies.modificacao.chamar_llm_json",
            _resposta_llm_requisitos(nota=10.0),
        )
        self._patch_llm(
            "tests.generator.chamar_llm_json",
            {"testes": [{"entrada": "5\n", "saida": "5\n", "obs": ""}]},
        )
        self._patch_llm("tests.generator.USAR_LLM", True)
        q = Questao(
            idx=8,
            tipo="modificacao",
            enunciado="Faça o programa repetir o valor digitado.",
            codigo_base="v = input()\nprint(v)",
            codigo_aluno_resposta="v = input()\nprint(v)",
        )
        res = estrategia_modificacao.avaliar(q)
        # Política 70/30 intacta.
        self.assertEqual(res.fonte_evidencia, "execucao")
        self.assertEqual(len(res.evidencias), 2)
        ev_exec, ev_llm = res.evidencias
        self.assertEqual(ev_exec["tipo"], "execucao")
        self.assertEqual(ev_exec["peso"], 0.7)
        self.assertEqual(ev_llm["tipo"], "llm")
        self.assertEqual(ev_llm["peso"], 0.3)
        # Procedência aparece naturalmente dentro da evidência de execução.
        self.assertEqual(
            ev_exec["dados"]["testes_por_origem"],
            {"enunciado": 0, "llm": 1},
        )


if __name__ == "__main__":
    unittest.main()
