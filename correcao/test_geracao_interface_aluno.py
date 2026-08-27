#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_geracao_interface_aluno.py

Testes de regressão para a geração de testes com interface baseada no
código do aluno (MODIFICACAO/CORRECAO).

Verifica que:
1. MODIFICACAO que adiciona input → interface detectada do código do aluno
2. MODIFICACAO que adiciona menu → prompts extraídos do código do aluno
3. MODIFICACAO que altera saída → enunciado define comportamento esperado
4. CORRECAO com interface igual → funciona normalmente
5. CORRECAO com saída diferente do base → testes refletem comportamento correto
6. Testes incompatíveis não são tratados como falhas (já testado, mas verifica novo fluxo)
7. Testes válidos avaliam comportamento solicitado pelo enunciado
8. LLM desativado → usa apenas testes explícitos
9. _codigo_para_interface retorna codigo_aluno para MODIFICACAO/CORRECAO
10. _codigo_para_interface retorna codigo_base para outros tipos
"""

from __future__ import annotations

import unittest
from unittest import mock

from models.questao import Questao
from tests.generator import (
    _codigo_para_interface,
    obter_testes,
    obter_testes_explicitos,
)


class CodigoParaInterfaceTests(unittest.TestCase):
    """Testa a função _codigo_para_interface()."""

    def test_modificacao_com_codigo_aluno(self):
        q = Questao(
            idx=1,
            tipo="modificacao",
            codigo_base="x = input()\nprint(x)",
            codigo_aluno_resposta='nome = input("Nome: ")\nidade = input("Idade: ")\nprint(nome, idade)',
        )
        interface = _codigo_para_interface(q)
        self.assertEqual(interface, q.codigo_aluno_resposta)

    def test_modificacao_sem_codigo_aluno_fallback_para_base(self):
        q = Questao(
            idx=1,
            tipo="modificacao",
            codigo_base="x = input()\nprint(x)",
            codigo_aluno_resposta="",
        )
        interface = _codigo_para_interface(q)
        self.assertEqual(interface, q.codigo_base)

    def test_correcao_com_codigo_aluno(self):
        q = Questao(
            idx=1,
            tipo="correcao",
            codigo_base="x = int(input())\nprint(x / 0)",
            codigo_aluno_resposta="x = int(input())\nprint(x / 2)",
        )
        interface = _codigo_para_interface(q)
        self.assertEqual(interface, q.codigo_aluno_resposta)

    def test_previsao_usa_codigo_base(self):
        q = Questao(
            idx=1,
            tipo="previsao",
            codigo_base="x = int(input())\nprint(x * 2)",
            codigo_aluno_resposta="5",
        )
        interface = _codigo_para_interface(q)
        self.assertEqual(interface, q.codigo_base)

    def test_descritiva_usa_codigo_base(self):
        q = Questao(
            idx=1,
            tipo="descritiva",
            codigo_base="print('hello')",
        )
        interface = _codigo_para_interface(q)
        self.assertEqual(interface, q.codigo_base)

    def test_codigo_aluno_none_fallback_para_base(self):
        q = Questao(
            idx=1,
            tipo="modificacao",
            codigo_base="x = input()\nprint(x)",
            codigo_aluno_resposta=None,
        )
        interface = _codigo_para_interface(q)
        self.assertEqual(interface, q.codigo_base)


class InterfaceDetectaInputTests(unittest.TestCase):
    """Verifica que a detecção de input() usa a interface do código do aluno."""

    @mock.patch("tests.generator.USAR_LLM", False)
    def test_modificacao_adiciona_inputdetectado(self):
        """
        Cenário 1: código_base NÃO tem input, código_aluno RESPOSTA tem input.
        → testes devem ser gerados (porque a interface atual tem input).
        """
        q = Questao(
            idx=1,
            tipo="modificacao",
            enunciado="Adicione input para ler o nome.",
            codigo_base="print('Olá')",
            codigo_aluno_resposta='nome = input("Nome: ")\nprint("Olá", nome)',
            entradaTestes="João\n",
            saida_esperada="Olá João\n",
        )
        testes = obter_testes(q)
        # Deve ter ao menos o teste explícito
        self.assertGreater(len(testes), 0)
        # O teste explícito deve ter entrada (porque a interface tem input)
        entradas = [t["entrada"] for t in testes]
        self.assertTrue(any(e.strip() for e in entradas))

    @mock.patch("tests.generator.USAR_LLM", False)
    def test_base_sem_input_aluno_com_input_gera_testes(self):
        """
        Cenário 2: código_base não tem input, código_aluno tem menu/input.
        → testes devem ser gerados.
        """
        q = Questao(
            idx=1,
            tipo="modificacao",
            enunciado="Adicione menu de operações.",
            codigo_base="a = int(input())\nb = int(input())\nprint(a + b)",
            codigo_aluno_resposta=(
                'a = int(input("A: "))\n'
                'b = int(input("B: "))\n'
                'op = input("Op (1=soma, 2=sub): ")\n'
                'if op == "1": print(a + b)\n'
                'elif op == "2": print(a - b)'
            ),
            entradaTestes="3\n5\n1\n",
            saida_esperada="8\n",
        )
        testes = obter_testes(q)
        self.assertGreater(len(testes), 0)
        # Interface detectada deve ter 3 inputs (do código do aluno)
        from utils.text import contar_inputs_codigo
        inputs_detectados = contar_inputs_codigo(_codigo_para_interface(q))
        self.assertEqual(inputs_detectados, 3)


class SaidaBaseadaEnunciadoTests(unittest.TestCase):
    """Verifica que a saída esperada é definida pelo enunciado, não pelo base."""

    @mock.patch("tests.generator.USAR_LLM", False)
    def test_modificacao_altera_saida_teste_reflete_enunciado(self):
        """
        Cenário 3: código_base imprime "Olá", enunciado pede "Bom dia".
        → teste explícito deve ter saída "Bom dia", não "Olá".
        """
        q = Questao(
            idx=1,
            tipo="modificacao",
            enunciado="Altere a saída para mostrar 'Bom dia' em vez de 'Olá'.",
            codigo_base="print('Olá')",
            codigo_aluno_resposta="print('Bom dia')",
            saida_esperada="Bom dia\n",
        )
        testes = obter_testes(q)
        self.assertGreater(len(testes), 0)
        saidas = [t["saida"] for t in testes]
        self.assertTrue(
            any("bom dia" in s.lower() for s in saidas),
            f"Saída deveria conter 'Bom dia', obtido: {saidas}",
        )

    @mock.patch("tests.generator.USAR_LLM", False)
    def test_correcao_saida_correta_diferente_do_base(self):
        """
        Cenário 5: código_base divide por zero, código corrigido divide normal.
        → teste deve esperar saída correta, não o erro do base.
        """
        q = Questao(
            idx=1,
            tipo="correcao",
            enunciado="Corrija o erro de divisão por zero.",
            codigo_base="x = int(input())\nprint(10 / 0)",
            codigo_aluno_resposta="x = int(input())\nprint(10 / x)",
            entradaTestes="2\n",
            saida_esperada="5.0\n",
        )
        testes = obter_testes(q)
        self.assertGreater(len(testes), 0)
        saidas = [t["saida"] for t in testes]
        self.assertTrue(
            any("5" in s for s in saidas),
            f"Saída deveria ser '5.0', obtido: {saidas}",
        )


class InterfaceIgualTests(unittest.TestCase):
    """Verifica que CORRECAO com interface igual funciona normalmente."""

    @mock.patch("tests.generator.USAR_LLM", False)
    def test_correcao_interface_igual_funciona(self):
        """
        Cenário 4: interface permanece igual, apenas lógica muda.
        → testes devem ser gerados corretamente.
        """
        q = Questao(
            idx=1,
            tipo="correcao",
            enunciado="Corrija o cálculo.",
            codigo_base="x = int(input())\nprint(x + 1)",  # errado
            codigo_aluno_resposta="x = int(input())\nprint(x * 2)",  # correto
            entradaTestes="3\n",
            saida_esperada="6\n",
        )
        testes = obter_testes(q)
        self.assertGreater(len(testes), 0)
        # Teste deve ter 1 entrada (1 input no código do aluno)
        from utils.text import contar_inputs_codigo
        inputs = contar_inputs_codigo(_codigo_para_interface(q))
        self.assertEqual(inputs, 1)


class LLMDesativadoTests(unittest.TestCase):
    """Verifica que com LLM desativado, apenas testes explícitos são usados."""

    @mock.patch("tests.generator.USAR_LLM", False)
    def test_llm_desativado_apenas_explicitos(self):
        q = Questao(
            idx=1,
            tipo="modificacao",
            enunciado="Adicione input.",
            codigo_base="print('ok')",
            codigo_aluno_resposta='x = input("X: ")\nprint(x)',
            entradaTestes="5\n",
            saida_esperada="5\n",
        )
        testes = obter_testes(q)
        # Sem LLM, apenas o teste explícito
        for t in testes:
            self.assertEqual(t.get("_origem"), "enunciado")


class IncompatibilidadeNoNovoFluxoTests(unittest.TestCase):
    """Verifica que testes incompatíveis continuam sendo tratados corretamente."""

    @mock.patch("evaluation.strategies.codigo.executar_codigo_python")
    def test_eof_error_input_adicional_e_incompativel(self, mock_exec):
        """
        Interface detectada do código do aluno (2 inputs), mas teste explícito
        fornece 1 entrada → incompatível.
        """
        from evaluation.strategies.codigo import avaliar

        def _executar(codigo, entrada, timeout=3):
            return {
                "stdout": "",
                "stderr": "Traceback...\nEOFError",
                "returncode": 1,
                "timeout": False,
                "erro_execucao": "Processo retornou código diferente de zero",
            }

        mock_exec.side_effect = _executar

        q = Questao(
            idx=1,
            tipo="modificacao",
            enunciado="Adicione idade.",
            codigo_base='nome = input("Nome: ")\nprint(nome)',
            codigo_aluno_resposta=(
                'nome = input("Nome: ")\n'
                'idade = input("Idade: ")\n'
                'print(nome, idade)'
            ),
            entradaTestes="João\n",
            saida_esperada="João ??\n",
        )

        testes = obter_testes(q)
        resultado = avaliar(q, q.codigo_aluno_resposta, testes)

        # Teste explícito com 1 entrada vs 2 inputs no código → incompatível
        incompativeis = sum(
            1 for t in resultado.testes_executados
            if t.get("compatibilidade") == "incompativel"
        )
        self.assertGreater(incompativeis, 0)
        # Nota não deve ser reduzida por incompatíveis
        self.assertEqual(resultado.nota, 0.0)  # 0 passou / 0 válidos


class PromptsExtraidosDoAlunoTests(unittest.TestCase):
    """Verifica que prompts de input() são extraídos do código do aluno."""

    @mock.patch("tests.generator.USAR_LLM", False)
    def test_limpeza_usa_prompts_do_aluno(self):
        """
        Quando o código do aluno tem prompts de input diferentes do base,
        a limpeza de entradas deve usar os prompts do aluno.
        """
        from tests.generator import limpar_entrada_interativa

        codigo_aluno = (
            'nome = input("Digite seu nome: ")\n'
            'print("Olá", nome)'
        )
        # Entrada com prompt do aluno
        entrada_bruta = "Digite seu nome: João\n"
        resultado = limpar_entrada_interativa(entrada_bruta, codigo_aluno)
        self.assertEqual(resultado.strip(), "João")

    @mock.patch("tests.generator.USAR_LLM", False)
    def test_limpeza_com_menu_do_aluno(self):
        """
        Quando o código do aluno tem menu de opções, a limpeza deve
        remover linhas de menu.
        """
        from tests.generator import limpar_entrada_interativa

        codigo_aluno = (
            'a = int(input("A: "))\n'
            'b = int(input("B: "))\n'
            'op = input("Op (1=soma, 2=sub): ")\n'
            'if op == "1": print(a + b)\n'
            'elif op == "2": print(a - b)'
        )
        entrada_bruta = (
            "A: 5\n"
            "B: 3\n"
            "1 - Soma\n"
            "2 - Subtração\n"
            "Op (1=soma, 2=sub): 1\n"
        )
        resultado = limpar_entrada_interativa(entrada_bruta, codigo_aluno)
        # Deve conter apenas os valores: 5, 3, 1
        linhas = [l.strip() for l in resultado.strip().split("\n") if l.strip()]
        self.assertEqual(linhas, ["5", "3", "1"])


if __name__ == "__main__":
    unittest.main()
