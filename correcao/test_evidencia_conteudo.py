#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_evidencia_conteudo.py

Etapa 4.3 — Rastreabilidade estruturada das evidências.

Valida que os caminhos ativos de avaliação registram em
Resultado.evidencias as evidências CONCRETAS que sustentaram a nota:

    a) codigo.py   → registro {"tipo": "execucao"} (ou lista vazia)
    b) previsao.py → registro "execucao" nos caminhos executados
    c) texto_llm   → registro "llm" completo (listas SEM truncamento,
       piso_aplicado explícito) ou "heuristica" no fallback
    d) modificacao → composição 70/30 representada por registros
       ponderados 0.7/0.3; só-LLM → registro único; placeholder → vazio
    e) validação administrativa → sempre evidencias == []
    f) invariante global:
           fonte_evidencia == "ausente"  <=>  evidencias == []
           fonte_evidencia != "ausente"  =>  existe evidência do tipo da fonte

O LLM é sempre mockado (unittest.mock); nenhum teste depende do LM Studio.
Nenhum campo existente é alterado; nenhuma lógica de nota é reexecutada.
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

from evaluation.dispatcher import corrigir_questao
from evaluation.evidencia import (
    FONTE_AUSENTE,
    FONTE_EXECUCAO,
    FONTE_HEURISTICA,
    FONTE_LLM,
    TIPOS_EVIDENCIA_VALIDOS,
    TIPO_EXECUCAO,
    TIPO_HEURISTICA,
    TIPO_LLM,
)
from evaluation.strategies import codigo as estrategia_codigo
from evaluation.strategies import modificacao as estrategia_modificacao
from evaluation.strategies import previsao as estrategia_previsao
from evaluation.strategies import texto_llm as estrategia_texto_llm
from models.questao import Questao
from report.formatter import formatar_resultado
from validation import (
    STATUS_ERRO_ENTRADA,
    STATUS_INCONCLUSIVO,
    STATUS_PENDENTE,
    validar_questao,
)


def _resposta_llm(
    nota: float = 8.0,
    status: str = "ok",
    acertos=None,
    melhorias=None,
) -> dict:
    return {
        "nota": nota,
        "status": status,
        "feedback": "boa resposta",
        "acertos": acertos if acertos is not None else ["conceito"],
        "melhorias": melhorias if melhorias is not None else [],
    }


def _resposta_llm_requisitos(nota: float = 10.0, status: str = "ok") -> dict:
    return {
        "nota": nota,
        "status": status,
        "cumpre_requisitos": True,
        "requisitos_identificados": ["imprimir 2", "usar print"],
        "requisitos_atendidos": ["imprimir 2", "usar print"],
        "faltantes": [],
        "feedback": "atende o pedido",
    }


class BaseConteudoTests(unittest.TestCase):
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


class CodigoEvidenciaTests(BaseConteudoTests):
    # a) codigo.py

    def test_com_testes_registra_execucao_com_contagens(self):
        q = Questao(idx=1, tipo="correcao", enunciado="Some dois números.")
        res = estrategia_codigo.avaliar(
            q, "print(1+1)", [{"entrada": "", "saida": "2\n"}]
        )
        self.assertEqual(res.status, "ok")
        self.assertEqual(len(res.evidencias), 1)
        ev = res.evidencias[0]
        self.assertEqual(ev["tipo"], TIPO_EXECUCAO)
        self.assertEqual(ev["dados"]["modo"], "com_testes")
        self.assertEqual(ev["dados"]["testes_total"], 1)
        self.assertEqual(ev["dados"]["testes_passaram"], 1)
        self.assertEqual(ev["dados"]["referencia"], "testes_executados")

    def test_sem_testes_modo_sem_testes(self):
        q = Questao(idx=1, tipo="correcao", enunciado="Imprima oi.")
        res = estrategia_codigo.avaliar(q, "print('oi')", [])
        self.assertEqual(len(res.evidencias), 1)
        ev = res.evidencias[0]
        self.assertEqual(ev["tipo"], TIPO_EXECUCAO)
        self.assertEqual(ev["dados"]["modo"], "sem_testes")
        self.assertFalse(ev["dados"]["erro_execucao"])

    def test_codigo_vazio_evidencias_vazias(self):
        q = Questao(idx=1, tipo="correcao", enunciado="Qualquer coisa.")
        res = estrategia_codigo.avaliar(q, "   ", [])
        self.assertEqual(res.evidencias, [])
        self.assertEqual(res.fonte_evidencia, FONTE_AUSENTE)

    def test_erro_sintaxe_registra_execucao(self):
        # O compilador Python foi executado sobre o código — evidência real.
        q = Questao(idx=1, tipo="correcao", enunciado="Defina uma função.")
        res = estrategia_codigo.avaliar(q, "def f(:", [])
        self.assertEqual(res.status, "erro")
        self.assertEqual(len(res.evidencias), 1)
        ev = res.evidencias[0]
        self.assertEqual(ev["tipo"], TIPO_EXECUCAO)
        self.assertEqual(ev["dados"]["modo"], "sintaxe")
        self.assertFalse(ev["dados"]["compilou"])

    def test_erro_runtime_registra_execucao_com_motivo(self):
        q = Questao(idx=1, tipo="correcao", enunciado="Divida por zero.")
        res = estrategia_codigo.avaliar(q, "x = 1/0", [])
        self.assertEqual(res.status, "erro")
        ev = res.evidencias[0]
        self.assertEqual(ev["tipo"], TIPO_EXECUCAO)
        self.assertTrue(ev["dados"]["erro_execucao"])
        self.assertTrue(ev["dados"]["motivo"])


class PrevisaoEvidenciaTests(BaseConteudoTests):
    # b) previsao.py

    def test_previsao_com_pares_registra_execucao(self):
        q = Questao(
            idx=2,
            tipo="previsao",
            enunciado="Qual será a saída?",
            codigo_base="n = input()\nprint(int(n) * 2)",
            entradaTestes="3\n",
            resposta_aluno="Entrada: 3\nSaída: 6",
        )
        res = estrategia_previsao.avaliar(q)
        self.assertEqual(res.status, "ok")
        self.assertEqual(len(res.evidencias), 1)
        ev = res.evidencias[0]
        self.assertEqual(ev["tipo"], TIPO_EXECUCAO)
        self.assertEqual(ev["dados"]["modo"], "com_pares")
        self.assertEqual(ev["dados"]["casos_total"], 1)
        self.assertEqual(ev["dados"]["casos_passaram"], 1)

    def test_previsao_legado_registra_execucao_com_similaridade(self):
        q = Questao(
            idx=2,
            tipo="previsao",
            enunciado="Qual será a saída?",
            codigo_base="print(7)",
            entradaTestes="",
            resposta_aluno="7",
        )
        res = estrategia_previsao.avaliar(q)
        self.assertEqual(len(res.evidencias), 1)
        ev = res.evidencias[0]
        self.assertEqual(ev["tipo"], TIPO_EXECUCAO)
        self.assertEqual(ev["dados"]["modo"], "legado")
        self.assertIsInstance(ev["dados"]["similaridade"], float)

    def test_previsao_sem_codigo_base_evidencias_vazias(self):
        q = Questao(idx=2, tipo="previsao", enunciado="", resposta_aluno="6")
        res = estrategia_previsao.avaliar(q)
        self.assertEqual(res.evidencias, [])
        self.assertEqual(res.fonte_evidencia, FONTE_AUSENTE)


class TextoLlmEvidenciaTests(BaseConteudoTests):
    # c) texto_llm.py

    def test_llm_valido_registro_completo_sem_truncamento(self):
        acertos = [f"acerto {i}" for i in range(1, 9)]      # 8 itens (> 5)
        melhorias = [f"melhoria {i}" for i in range(1, 7)]  # 6 itens (> 5)
        self._patch_llm(
            "evaluation.strategies.texto_llm.chamar_llm_json",
            _resposta_llm(nota=8.0, status="ok", acertos=acertos, melhorias=melhorias),
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
        self.assertEqual(len(res.evidencias), 1)
        dados = res.evidencias[0]["dados"]
        self.assertEqual(dados["nota_original"], 8.0)
        self.assertEqual(dados["status"], "ok")
        # Listas preservadas INTEGRALMENTE na evidência (sem corte de exibição).
        self.assertEqual(dados["acertos"], acertos)
        self.assertEqual(dados["melhorias"], melhorias)
        self.assertIs(dados["piso_aplicado"], False)

    def test_piso_aplicado_registrado_explicitamente(self):
        self._patch_llm(
            "evaluation.strategies.texto_llm.chamar_llm_json",
            _resposta_llm(nota=4.0, status="ok"),
        )
        q = Questao(
            idx=3,
            tipo="justificativa",
            enunciado="Por que padronizamos o uso de minusculas e maiusculas?",
            rubrica="Reconhece diferença entre caixas.",
            resposta_aluno="Para diferenciar identificadores, pois Python compara lower e upper.",
        )
        res = estrategia_texto_llm.avaliar(q)
        # Política do piso intacta: 4.0 elevado para 7.0.
        self.assertEqual(res.nota, 7.0)
        self.assertEqual(res.fonte_evidencia, FONTE_LLM)
        dados = res.evidencias[0]["dados"]
        self.assertIs(dados["piso_aplicado"], True)
        self.assertEqual(dados["nota_original"], 4.0)
        self.assertIn("piso", res.evidencias[0]["resumo"].lower())

    def test_fallback_heuristico_registra_motivo(self):
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
        self.assertEqual(len(res.evidencias), 1)
        ev = res.evidencias[0]
        self.assertEqual(ev["tipo"], TIPO_HEURISTICA)
        self.assertEqual(ev["dados"]["motivo"], "llm_sem_json_valido")
        self.assertIs(ev["dados"]["conceito_ok"], False)

    def test_resposta_vazia_evidencias_vazias(self):
        q = Questao(idx=3, tipo="descritiva", enunciado="Explique algo.", rubrica="r")
        res = estrategia_texto_llm.avaliar(q)
        self.assertEqual(res.evidencias, [])
        self.assertEqual(res.fonte_evidencia, FONTE_AUSENTE)


class ModificacaoEvidenciaTests(BaseConteudoTests):
    # d) modificacao.py — política 70/30 PRESERVADA

    def test_com_testes_duas_evidencias_ponderadas(self):
        self._patch_llm(
            "evaluation.strategies.modificacao.chamar_llm_json",
            _resposta_llm_requisitos(nota=10.0, status="ok"),
        )
        self._patch_llm("evaluation.strategies.modificacao.USAR_LLM", True)
        q = Questao(
            idx=5,
            tipo="modificacao",
            enunciado="Faça o programa imprimir 2.",
            codigo_base="print(1)",
            codigo_aluno_resposta="print(2)",
            testes=[{"entrada": "", "saida": "2\n", "obs": ""}],
        )
        res = estrategia_modificacao.avaliar(q)
        # Política 4.2 intacta: fonte predominante continua "execucao".
        self.assertEqual(res.fonte_evidencia, FONTE_EXECUCAO)
        self.assertEqual(res.nota, 10.0)  # 0.7*10 + 0.3*10
        self.assertEqual(len(res.evidencias), 2)

        ev_exec, ev_llm = res.evidencias
        self.assertEqual(ev_exec["tipo"], TIPO_EXECUCAO)
        self.assertEqual(ev_exec["peso"], 0.7)
        self.assertEqual(ev_exec["dados"]["modo"], "com_testes")
        self.assertEqual(ev_llm["tipo"], TIPO_LLM)
        self.assertEqual(ev_llm["peso"], 0.3)
        # Listas do LLM preservadas integralmente.
        self.assertEqual(
            ev_llm["dados"]["requisitos_atendidos"],
            ["imprimir 2", "usar print"],
        )

    def test_somente_llm_um_registro_sem_peso(self):
        self._patch_llm(
            "evaluation.strategies.modificacao.chamar_llm_json",
            _resposta_llm_requisitos(nota=9.0, status="parcial"),
        )
        self._patch_llm("evaluation.strategies.modificacao.USAR_LLM", True)
        q = Questao(
            idx=5,
            tipo="modificacao",
            enunciado="Adicione um comentário ao programa.",
            codigo_base="x = 1",
            codigo_aluno_resposta="x = 1  # comentário adicionado",
        )
        res = estrategia_modificacao.avaliar(q)
        self.assertEqual(res.fonte_evidencia, FONTE_LLM)
        self.assertEqual(res.nota, 9.0)
        self.assertEqual(len(res.evidencias), 1)
        ev = res.evidencias[0]
        self.assertEqual(ev["tipo"], TIPO_LLM)
        self.assertNotIn("peso", ev)  # peso só existe quando há composição

    def test_placeholder_sem_testes_e_sem_llm_evidencias_vazias(self):
        q = Questao(
            idx=5,
            tipo="modificacao",
            enunciado="Modifique o programa para ler dois valores.",
            codigo_base="a = input()\nprint(a)",
            codigo_aluno_resposta="a = input()\nb = input()\nprint(a, b)",
        )
        with mock.patch("evaluation.strategies.modificacao.USAR_LLM", False):
            res = estrategia_modificacao.avaliar(q)
        self.assertEqual(res.status, "parcial")
        self.assertEqual(res.nota, 0.0)
        self.assertEqual(res.evidencias, [])
        self.assertEqual(res.fonte_evidencia, FONTE_AUSENTE)

    def test_llm_desativado_herde_evidencia_da_parte_objetiva(self):
        q = Questao(
            idx=5,
            tipo="modificacao",
            enunciado="Faça o programa imprimir 2.",
            codigo_base="print(1)",
            codigo_aluno_resposta="print(2)",
            testes=[{"entrada": "", "saida": "2\n", "obs": ""}],
        )
        with mock.patch("evaluation.strategies.modificacao.USAR_LLM", False):
            res = estrategia_modificacao.avaliar(q)
        self.assertEqual(res.fonte_evidencia, FONTE_EXECUCAO)
        self.assertEqual(len(res.evidencias), 1)
        ev = res.evidencias[0]
        self.assertEqual(ev["tipo"], TIPO_EXECUCAO)
        self.assertNotIn("peso", ev)  # herdada sem composição, sem peso


class ValidacaoAdministrativaEvidenciaTests(unittest.TestCase):
    # e) resultados administrativos nunca carregam evidências

    def test_erro_entrada_evidencias_vazias(self):
        q = Questao(idx=4, tipo="previsao", enunciado="")
        res = validar_questao(q)
        self.assertIsNotNone(res)
        self.assertEqual(res.status, STATUS_ERRO_ENTRADA)
        self.assertEqual(res.evidencias, [])

    def test_pendente_evidencias_vazias(self):
        q = Questao(
            idx=4,
            tipo="justificativa",
            enunciado="Justifique o uso de listas.",
            rubrica="Menciona ordenação.",
        )
        res = validar_questao(q)
        self.assertIsNotNone(res)
        self.assertEqual(res.status, STATUS_PENDENTE)
        self.assertEqual(res.evidencias, [])

    def test_inconclusivo_evidencias_vazias(self):
        q = Questao(
            idx=4,
            tipo="correcao",
            enunciado="Corrija o código abaixo.",
            codigo_base="x = 1",
            codigo_aluno_resposta="x = 2",
        )
        res = validar_questao(q)
        self.assertIsNotNone(res)
        self.assertEqual(res.status, STATUS_INCONCLUSIVO)
        self.assertEqual(res.evidencias, [])


class InvarianteFonteEvidenciasTests(BaseConteudoTests):
    # f) fonte_evidencia == "ausente" <=> evidencias == []

    def _cenarios(self) -> dict:
        cenarios: dict = {}

        cenarios["codigo_com_testes"] = estrategia_codigo.avaliar(
            Questao(idx=1, tipo="correcao", enunciado="Some."),
            "print(1+1)",
            [{"entrada": "", "saida": "2\n"}],
        )
        cenarios["codigo_vazio"] = estrategia_codigo.avaliar(
            Questao(idx=1, tipo="correcao", enunciado="Qualquer."), "   ", []
        )
        cenarios["previsao_pares"] = estrategia_previsao.avaliar(
            Questao(
                idx=2, tipo="previsao", enunciado="Saída?",
                codigo_base="print(7)", entradaTestes="", resposta_aluno="Entrada: \nSaída: 7",
            )
        )
        cenarios["previsao_sem_codigo"] = estrategia_previsao.avaliar(
            Questao(idx=2, tipo="previsao", enunciado="", resposta_aluno="6")
        )
        cenarios["texto_vazio"] = estrategia_texto_llm.avaliar(
            Questao(idx=3, tipo="descritiva", enunciado="Explique algo.", rubrica="r")
        )

        admin = validar_questao(Questao(idx=4, tipo="previsao", enunciado=""))
        if admin is not None:
            cenarios["admin_erro_entrada"] = admin

        return cenarios

    def _cenarios_com_llm(self) -> dict:
        cenarios: dict = {}

        self._patch_llm(
            "evaluation.strategies.texto_llm.chamar_llm_json", _resposta_llm()
        )
        cenarios["texto_llm_ok"] = estrategia_texto_llm.avaliar(
            Questao(
                idx=3, tipo="descritiva",
                enunciado="Explique o que é uma variável.",
                rubrica="Armazenamento.",
                resposta_aluno="Local que armazena um valor.",
            )
        )

        self._patch_llm(
            "evaluation.strategies.modificacao.chamar_llm_json",
            _resposta_llm_requisitos(nota=10.0),
        )
        self._patch_llm("evaluation.strategies.modificacao.USAR_LLM", True)
        cenarios["mod_7030"] = estrategia_modificacao.avaliar(
            Questao(
                idx=5, tipo="modificacao",
                enunciado="Faça o programa imprimir 2.",
                codigo_base="print(1)", codigo_aluno_resposta="print(2)",
                testes=[{"entrada": "", "saida": "2\n", "obs": ""}],
            )
        )
        cenarios["mod_so_llm"] = estrategia_modificacao.avaliar(
            Questao(
                idx=5, tipo="modificacao",
                enunciado="Adicione um comentário ao programa.",
                codigo_base="x = 1", codigo_aluno_resposta="x = 1  # comentário",
            )
        )

        return cenarios

    def test_invariante_global_em_todos_os_caminhos(self):
        self._patch_llm("evaluation.strategies.texto_llm.chamar_llm_json", None)
        cenarios = self._cenarios()

        cenarios["fallback_heuristico"] = estrategia_texto_llm.avaliar(
            Questao(
                idx=3, tipo="justificativa",
                enunciado="Justifique o uso de funções.",
                rubrica="Organização.", resposta_aluno="Porque sim.",
            )
        )

        with mock.patch("evaluation.strategies.modificacao.USAR_LLM", False):
            cenarios["placeholder_mod"] = estrategia_modificacao.avaliar(
                Questao(
                    idx=5, tipo="modificacao",
                    enunciado="Leia dois valores.",
                    codigo_base="a = input()\nprint(a)",
                    codigo_aluno_resposta="a = input()\nb = input()\nprint(a, b)",
                )
            )

        pendente = corrigir_questao(
            Questao(
                idx=6, tipo="descritiva",
                enunciado="Descreva o que faz uma fila.", rubrica="FIFO.",
            )
        )
        cenarios["dispatcher_pendente"] = pendente

        cenarios.update(self._cenarios_com_llm())

        self.assertGreaterEqual(len(cenarios), 12)
        for nome, res in cenarios.items():
            with self.subTest(cenario=nome):
                for ev in res.evidencias:
                    self.assertIn(ev.get("tipo"), TIPOS_EVIDENCIA_VALIDOS)
                if res.fonte_evidencia == FONTE_AUSENTE:
                    self.assertEqual(
                        res.evidencias, [],
                        msg=f"{nome}: fonte ausente exige evidencias vazias",
                    )
                else:
                    self.assertTrue(
                        res.evidencias,
                        msg=f"{nome}: fonte '{res.fonte_evidencia}' exige evidência",
                    )
                    tipos = {ev.get("tipo") for ev in res.evidencias}
                    self.assertIn(
                        res.fonte_evidencia,
                        tipos,
                        msg=f"{nome}: nenhuma evidência compatível com a fonte",
                    )


class EvidenciaSerializavelTests(BaseConteudoTests):
    # Os registros devem ser JSON-serializáveis (contrato do schema).

    def test_evidencias_da_composicao_sao_json_serializaveis(self):
        self._patch_llm(
            "evaluation.strategies.modificacao.chamar_llm_json",
            _resposta_llm_requisitos(nota=10.0),
        )
        self._patch_llm("evaluation.strategies.modificacao.USAR_LLM", True)
        res = estrategia_modificacao.avaliar(
            Questao(
                idx=5, tipo="modificacao",
                enunciado="Faça o programa imprimir 2.",
                codigo_base="print(1)", codigo_aluno_resposta="print(2)",
                testes=[{"entrada": "", "saida": "2\n", "obs": ""}],
            )
        )
        serializado = json.dumps(res.evidencias, ensure_ascii=False)
        self.assertIn("execucao", serializado)
        self.assertIn("llm", serializado)

    def test_evidencia_do_llm_textual_e_json_serializavel(self):
        self._patch_llm(
            "evaluation.strategies.texto_llm.chamar_llm_json",
            _resposta_llm(acertos=["a", "b"], melhorias=["c"]),
        )
        res = estrategia_texto_llm.avaliar(
            Questao(
                idx=3, tipo="descritiva",
                enunciado="Explique o que é uma variável.",
                rubrica="Armazenamento.",
                resposta_aluno="Local que armazena um valor.",
            )
        )
        json.dumps(res.evidencias, ensure_ascii=False)


class FormatterEvidenciasTests(BaseConteudoTests):
    # Seção de evidências aparece só quando há registros.

    def test_relatorio_exibe_secao_quando_ha_evidencias(self):
        res = estrategia_codigo.avaliar(
            Questao(idx=1, tipo="correcao", enunciado="Some."),
            "print(1+1)",
            [{"entrada": "", "saida": "2\n"}],
        )
        texto = formatar_resultado(res, Questao(idx=1, tipo="correcao"))
        self.assertIn("Evidências:", texto)
        self.assertIn("execucao", texto)
        self.assertIn("referencia", texto)

    def test_relatorio_nao_exibe_secao_vazia(self):
        q = Questao(idx=1, tipo="descritiva", enunciado="Explique algo.", rubrica="r")
        res = estrategia_texto_llm.avaliar(q)  # resposta vazia → sem evidências
        texto = formatar_resultado(res, q)
        self.assertNotIn("Evidências:", texto)


if __name__ == "__main__":
    unittest.main()
