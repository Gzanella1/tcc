#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_interface_dinamica.py

Testes para a detecção e geração de testes em códigos com interface
DINÂMICA: input() dentro de um laço (for/while), onde a quantidade de
chamadas em tempo de execução não é fixa e não pode ser validada por
contagem estática de assinatura.

Cobre:
    1. Detecção de input() dentro de loop (codigo_tem_input_em_loop).
    2. Geração de testes aceita para código com loop (não retorna mais
       [] sistematicamente, como acontecia antes da mudança).
    3. Validação por EXECUÇÃO REAL: entradas mecanicamente incompletas
       (faltando a linha que encerra o loop) são descartadas; entradas
       completas são aceitas.

Os testes da seção 3 (ValidacaoPorExecucaoTests) executam código de
verdade via execution/runner.py (container Docker/Podman). Se o runtime
de container não estiver disponível no ambiente, eles são pulados
automaticamente (mesma lógica de test_runner_security.py).
"""

from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from config import CONTAINER_IMAGE
from execution import runner
from models.questao import Questao
from tests.generator import (
    _gerar_testes_llm_once,
    obter_testes,
)
from utils.text import codigo_tem_input_em_loop


def _runtime_disponivel() -> str | None:
    runtime = runner._resolver_runtime()
    if not runtime:
        return None
    try:
        proc = subprocess.run(
            [runtime, "image", "inspect", CONTAINER_IMAGE],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        return None
    return runtime if proc.returncode == 0 else None


RUNTIME = _runtime_disponivel()


# ─── 1. Detecção de loop ──────────────────────────────────────────────────────

class DeteccaoLoopTests(unittest.TestCase):
    def test_input_dentro_de_while_e_detectado(self):
        codigo = (
            'while True:\n'
            '    chave = input("chave: ")\n'
            '    if chave == "":\n'
            '        break\n'
        )
        self.assertTrue(codigo_tem_input_em_loop(codigo))

    def test_input_fora_de_loop_nao_e_detectado(self):
        codigo = 'nome = input("Nome: ")\nprint(nome)'
        self.assertFalse(codigo_tem_input_em_loop(codigo))

    def test_input_dentro_de_for_e_detectado(self):
        codigo = (
            'for i in range(3):\n'
            '    x = input()\n'
        )
        self.assertTrue(codigo_tem_input_em_loop(codigo))

    def test_codigo_vazio_nao_e_detectado(self):
        self.assertFalse(codigo_tem_input_em_loop(""))

    def test_codigo_com_syntax_error_nao_quebra(self):
        codigo = "while True input('x')"
        self.assertFalse(codigo_tem_input_em_loop(codigo))


# ─── 2. Geração aceita para código com loop ──────────────────────────────────

CODIGO_BUSCA = (
    'def busca_por_chave(d1, chave):\n'
    '    return d1.get(chave)\n'
    'def main():\n'
    '    d1 = {}\n'
    '    while True:\n'
    '        chave = input("Digite uma chave: ")\n'
    '        if chave == "":\n'
    '            break\n'
    '        valor = input("Digite um valor: ")\n'
    '        d1[chave] = valor\n'
    '    chave = input("Digite a chave para buscar o valor: ")\n'
    '    valor = busca_por_chave(d1, chave)\n'
    '    print(valor)\n'
    'main()\n'
)


class GeracaoComInterfaceDinamicaTests(unittest.TestCase):
    """Garante que código com loop não é mais descartado sistematicamente
    pela contagem estática de assinatura (comportamento antigo)."""

    @mock.patch("tests.generator.USAR_LLM", False)
    def test_sem_llm_interface_dinamica_nao_quebra_fluxo_explicito(self):
        """Com LLM desligado, testes explícitos continuam funcionando
        normalmente mesmo com código de interface dinâmica."""
        q = Questao(
            idx=1,
            tipo="modificacao",
            enunciado="Busque valores por chave.",
            codigo_base="d = {}\nprint(d)",
            codigo_aluno_resposta=CODIGO_BUSCA,
            entradaTestes="a\n1\n\na\n",
            saida_esperada="1\n",
        )
        testes = obter_testes(q)
        self.assertEqual(len(testes), 1)
        self.assertEqual(testes[0]["_origem"], "enunciado")


# ─── 3. Validação por execução real (requer container) ──────────────────────

@unittest.skipUnless(
    RUNTIME,
    f"Docker/Podman indisponível ou imagem {CONTAINER_IMAGE!r} ausente localmente",
)
class ValidacaoPorExecucaoTests(unittest.TestCase):
    """Confirma que entradas mecanicamente incompletas (EOFError) são
    descartadas, mesmo com interface dinâmica, e que entradas completas
    são aceitas. Executa código real via container."""

    @mock.patch("tests.generator.USAR_LLM", True)
    @mock.patch("tests.generator.chamar_llm_json")
    def test_entrada_sem_terminador_e_descartada(self, mock_llm):
        mock_llm.return_value = {
            "testes": [
                {
                    "entrada": "chave1\nvalor1\nchave2\nvalor2\nchave1",
                    "saida": "valor1\n",
                    "obs": "sem terminador do loop",
                }
            ]
        }
        q = Questao(
            idx=1,
            tipo="modificacao",
            enunciado="Busque valores por chave.",
            codigo_base="d = {}\nprint(d)",
            codigo_aluno_resposta=CODIGO_BUSCA,
        )
        testes = _gerar_testes_llm_once(q, quantidade=1)
        self.assertEqual(testes, [])

    @mock.patch("tests.generator.USAR_LLM", True)
    @mock.patch("tests.generator.chamar_llm_json")
    def test_entrada_com_terminador_e_aceita(self, mock_llm):
        mock_llm.return_value = {
            "testes": [
                {
                    "entrada": "chave1\nvalor1\n\nchave1",
                    "saida": "valor1\n",
                    "obs": "com terminador do loop",
                }
            ]
        }
        q = Questao(
            idx=1,
            tipo="modificacao",
            enunciado="Busque valores por chave.",
            codigo_base="d = {}\nprint(d)",
            codigo_aluno_resposta=CODIGO_BUSCA,
        )
        testes = _gerar_testes_llm_once(q, quantidade=1)
        self.assertEqual(len(testes), 1)
        self.assertEqual(testes[0]["_origem"], "llm")


# ─── 3b. Mesma validação, mockando execucao (não depende de container) ──────

class ValidacaoPorExecucaoMockadaTests(unittest.TestCase):

    @mock.patch("tests.generator.USAR_LLM", True)
    @mock.patch("tests.generator.executar_codigo_python")
    @mock.patch("tests.generator.chamar_llm_json")
    def test_entrada_incompleta_gera_eof_e_e_descartada(self, mock_llm, mock_exec):
        mock_llm.return_value = {
            "testes": [
                {
                    "entrada": "chave1\nvalor1\nchave2\nvalor2\nchave1",
                    "saida": "valor1\n",
                    "obs": "sem terminador do loop",
                }
            ]
        }
        mock_exec.return_value = {
            "stdout": "",
            "stderr": (
                "Traceback (most recent call last):\n"
                '  File "resposta_aluno.py", line 11, in <module>\n'
                "    valor = input(\"Digite um valor: \")\n"
                "EOFError"
            ),
            "returncode": 1,
            "timeout": False,
            "erro_execucao": "Processo retornou código diferente de zero",
        }

        q = Questao(
            idx=1,
            tipo="modificacao",
            enunciado="Busque valores por chave.",
            codigo_base="d = {}\nprint(d)",
            codigo_aluno_resposta=CODIGO_BUSCA,
        )
        testes = _gerar_testes_llm_once(q, quantidade=1)
        self.assertEqual(testes, [])

    @mock.patch("tests.generator.USAR_LLM", True)
    @mock.patch("tests.generator.executar_codigo_python")
    @mock.patch("tests.generator.chamar_llm_json")
    def test_entrada_completa_executa_sem_erro_e_e_aceita(self, mock_llm, mock_exec):
        mock_llm.return_value = {
            "testes": [
                {
                    "entrada": "chave1\nvalor1\n\nchave1",
                    "saida": "valor1\n",
                    "obs": "com terminador do loop",
                }
            ]
        }
        mock_exec.return_value = {
            "stdout": "valor1\n",
            "stderr": "",
            "returncode": 0,
            "timeout": False,
            "erro_execucao": "",
        }

        q = Questao(
            idx=1,
            tipo="modificacao",
            enunciado="Busque valores por chave.",
            codigo_base="d = {}\nprint(d)",
            codigo_aluno_resposta=CODIGO_BUSCA,
        )
        testes = _gerar_testes_llm_once(q, quantidade=1)
        self.assertEqual(len(testes), 1)
        self.assertEqual(testes[0]["_origem"], "llm")

    @mock.patch("tests.generator.USAR_LLM", True)
    @mock.patch("tests.generator.executar_codigo_python")
    @mock.patch("tests.generator.chamar_llm_json")
    def test_timeout_na_validacao_descarta_teste(self, mock_llm, mock_exec):
        mock_llm.return_value = {
            "testes": [
                {
                    "entrada": "chave1\nvalor1\n\nchave1",
                    "saida": "valor1\n",
                    "obs": "trava em loop infinito por algum motivo",
                }
            ]
        }
        mock_exec.return_value = {
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "timeout": True,
            "erro_execucao": "Timeout",
        }

        q = Questao(
            idx=1,
            tipo="modificacao",
            enunciado="Busque valores por chave.",
            codigo_base="d = {}\nprint(d)",
            codigo_aluno_resposta=CODIGO_BUSCA,
        )
        testes = _gerar_testes_llm_once(q, quantidade=1)
        self.assertEqual(testes, [])


if __name__ == "__main__":
    unittest.main()