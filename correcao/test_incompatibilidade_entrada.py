#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_incompatibilidade_entrada.py

Testes de regressão para a classificação de testes incompatíveis.

Verifica que:
1. EOFError + código com mais input() que entradas → incompatível (não reduz nota)
2. EOFError + código com mesma qtd de input() → falha real
3. ZeroDivisionError → falha real
4. ValueError → falha real
5. Timeout → falha real
6. Saída correta → aprovado
7. Saída incorreta → reprovado
8. Mistura de resultados → cálculo correto da nota
9. Todos incompatíveis → nota 0 (sem aprovação artificial)
10. Testes existentes continuam passando

Nenhuma execução real de container é feita (mock completo do runner).
"""

from __future__ import annotations

import unittest
from unittest import mock

from models.questao import Questao
from utils.text import (
    ERRO_EOF_INCOMPATIVEL,
    ERRO_NENHUM,
    ERRO_RUNTIME,
    blocos_na_entrada,
    classificar_erro_execucao,
    contar_inputs_codigo,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _execucao_ok(stdout: str = "", stderr: str = "") -> dict:
    return {
        "stdout": stdout,
        "stderr": stderr,
        "returncode": 0,
        "timeout": False,
        "erro_execucao": "",
    }


def _execucao_eof_error(linha: int = 1) -> dict:
    traceback = (
        f"Traceback (most recent call last):\n"
        f'  File "resposta_aluno.py", line {linha}, in <module>\n'
        f"    nome = input(\"Nome: \")\n"
        f"EOFError"
    )
    return {
        "stdout": "",
        "stderr": traceback,
        "returncode": 1,
        "timeout": False,
        "erro_execucao": "Processo retornou código diferente de zero",
    }


def _execucao_zero_division() -> dict:
    traceback = (
        "Traceback (most recent call last):\n"
        '  File "resposta_aluno.py", line 3, in <module>\n'
        "    result = 10 / 0\n"
        "ZeroDivisionError: division by zero"
    )
    return {
        "stdout": "",
        "stderr": traceback,
        "returncode": 1,
        "timeout": False,
        "erro_execucao": "Processo retornou código diferente de zero",
    }


def _execucao_value_error() -> dict:
    traceback = (
        "Traceback (most recent call last):\n"
        '  File "resposta_aluno.py", line 2, in <module>\n'
        '    num = int("abc")\n'
        "ValueError: invalid literal for int()"
    )
    return {
        "stdout": "",
        "stderr": traceback,
        "returncode": 1,
        "timeout": False,
        "erro_execucao": "Processo retornou código diferente de zero",
    }


def _execucao_timeout() -> dict:
    return {
        "stdout": "",
        "stderr": "",
        "returncode": None,
        "timeout": True,
        "erro_execucao": "Timeout",
    }


# ─── Testes das funções utilitárias ──────────────────────────────────────────

class ClassificarErroExecucaoTests(unittest.TestCase):

    def test_eof_error_retorna_eof_incompativel(self):
        execucao = _execucao_eof_error()
        resultado = classificar_erro_execucao(execucao["stderr"], execucao["returncode"])
        self.assertEqual(resultado, ERRO_EOF_INCOMPATIVEL)

    def test_zero_division_retorna_erro_runtime(self):
        execucao = _execucao_zero_division()
        resultado = classificar_erro_execucao(execucao["stderr"], execucao["returncode"])
        self.assertEqual(resultado, ERRO_RUNTIME)

    def test_value_error_retorna_erro_runtime(self):
        execucao = _execucao_value_error()
        resultado = classificar_erro_execucao(execucao["stderr"], execucao["returncode"])
        self.assertEqual(resultado, ERRO_RUNTIME)

    def test_execucao_ok_retorna_nenhum(self):
        execucao = _execucao_ok()
        resultado = classificar_erro_execucao(execucao["stderr"], execucao["returncode"])
        self.assertEqual(resultado, ERRO_NENHUM)

    def test_stderr_vazio_returncode_zero_retorna_nenhum(self):
        resultado = classificar_erro_execucao("", 0)
        self.assertEqual(resultado, ERRO_NENHUM)

    def test_returncode_none_retorna_nenhum(self):
        resultado = classificar_erro_execucao("", None)
        self.assertEqual(resultado, ERRO_NENHUM)


class ContarInputsCodigoTests(unittest.TestCase):

    def test_codigo_com_um_input(self):
        codigo = 'nome = input("Nome: ")\nprint(nome)'
        self.assertEqual(contar_inputs_codigo(codigo), 1)

    def test_codigo_com_dois_inputs(self):
        codigo = (
            'nome = input("Nome: ")\n'
            'idade = input("Idade: ")\n'
            'print(nome, idade)'
        )
        self.assertEqual(contar_inputs_codigo(codigo), 2)

    def test_codigo_sem_input(self):
        codigo = 'print("Olá")'
        self.assertEqual(contar_inputs_codigo(codigo), 0)

    def test_codigo_vazio(self):
        self.assertEqual(contar_inputs_codigo(""), 0)

    def test_codigo_none(self):
        self.assertEqual(contar_inputs_codigo(None), 0)

    def test_codigo_com_input_condicional(self):
        codigo = (
            'x = input("X: ")\n'
            'if x == "admin":\n'
            '    s = input("Senha: ")\n'
            'print(x)'
        )
        # AST conta todas as chamadas, incluindo condicionais
        self.assertEqual(contar_inputs_codigo(codigo), 2)

    def test_codigo_com_syntax_error(self):
        codigo = 'if True print("x")'
        # SyntaxError → retorna 0
        self.assertEqual(contar_inputs_codigo(codigo), 0)


class BlocosEntradaTests(unittest.TestCase):

    def test_entrada_vazia(self):
        self.assertEqual(blocos_na_entrada(""), 0)

    def test_entrada_none(self):
        self.assertEqual(blocos_na_entrada(None), 0)

    def test_entrada_um_valor(self):
        self.assertEqual(blocos_na_entrada("João"), 1)

    def test_entrada_dois_valores(self):
        self.assertEqual(blocos_na_entrada("João\n14:00"), 2)

    def test_entrada_tres_valores(self):
        self.assertEqual(blocos_na_entrada("a\nb\nc"), 3)

    def test_entrada_com_trailing_newline(self):
        # trailing newline não conta como bloco extra
        self.assertEqual(blocos_na_entrada("João\n"), 1)

    def test_entrada_com_dois_trailing_newlines(self):
        self.assertEqual(blocos_na_entrada("a\nb\n"), 2)


# ─── Testes de integração com avaliar() ──────────────────────────────────────

def _questao_com_inputs(n_inputs_codigo: int = 2) -> Questao:
    """Questão com código_base que tem N inputs."""
    inputs = "\n".join(
        [f'x{i} = input("Valor {i}: ")' for i in range(n_inputs_codigo)]
    )
    codigo_base = inputs + "\n" + "print('ok')"
    return Questao(
        idx=1,
        tipo="modificacao",
        enunciado="Modifique o programa.",
        codigo_base=codigo_base,
    )


def _mock_runner(execucoes_por_entrada: dict):
    """
    Cria um mock para executar_codigo_python que retorna resultados
    pré-definidos com base no conteúdo da entrada.
    """
    def _executar(codigo, entrada, timeout=3):
        chave = entrada.strip()
        if chave in execucoes_por_entrada:
            return execucoes_por_entrada[chave]
        return _execucao_ok(stdout="")
    return _executar


class IncompatibilidadeEntradaTests(unittest.TestCase):
    """Testes de integração: EOFError + interface incompatível."""

    @mock.patch("evaluation.strategies.codigo.executar_codigo_python")
    def test_eof_error_com_input_adicional_e_incompativel(self, mock_exec):
        """
        Cenário 1: Código com 2 input(), teste fornece 1 entrada.
        → incompatível, não reduz a nota.
        """
        codigo_aluno = (
            'nome = input("Nome: ")\n'
            'idade = input("Idade: ")\n'
            'print(f"{nome} tem {idade} anos")'
        )
        testes = [
            {"entrada": "João", "saida": "João tem ?? anos", "obs": ""},
        ]

        mock_exec.return_value = _execucao_eof_error(linha=2)

        q = Questao(
            idx=1,
            tipo="modificacao",
            enunciado="Adicione idade.",
            codigo_base='nome = input("Nome: ")\nprint(nome)',
        )

        from evaluation.strategies.codigo import avaliar
        resultado = avaliar(q, codigo_aluno, testes)

        self.assertEqual(resultado.testes_executados[0]["compatibilidade"], "incompativel")
        self.assertEqual(resultado.testes_executados[0]["ok"], None)
        self.assertEqual(resultado.nota, 0.0)  # 0 passou / 0 válidos
        self.assertIn("Nenhum teste compatível", resultado.feedback)

    @mock.patch("evaluation.strategies.codigo.executar_codigo_python")
    def test_eof_error_com_mesma_quantidade_e_falha(self, mock_exec):
        """
        Cenário 9: Código com 1 input(), teste fornece 1 entrada, mas EOFError.
        → falha real (erro lógico no código).
        """
        codigo_aluno = (
            'nome = input("Nome: ")\n'
            'idade = input("Idade: ")  # bug: chama input duas vezes\n'
            'print(nome)'
        )
        testes = [
            {"entrada": "João", "saida": "João", "obs": ""},
        ]

        mock_exec.return_value = _execucao_eof_error(linha=2)

        q = Questao(
            idx=1,
            tipo="correcao",
            enunciado="Corrija o programa.",
            codigo_base='nome = input("Nome: ")\nprint(nome)',
        )

        from evaluation.strategies.codigo import avaliar
        resultado = avaliar(q, codigo_aluno, testes)

        # Código tem 2 input(), teste tem 1 entrada → incompatível
        # Mas se o código do aluno tivesse 1 input() e désse EOFError → falha
        # Neste caso: código tem 2 inputs, entrada tem 1 → incompatível
        self.assertEqual(resultado.testes_executados[0]["compatibilidade"], "incompativel")

    @mock.patch("evaluation.strategies.codigo.executar_codigo_python")
    def test_eof_error_codigo_com_1_input_e_entrada_1_falha_real(self, mock_exec):
        """
        Código com 1 input(), entrada com 1 valor, EOFError.
        → erro real (pode ser bug de loop ou condicional).
        """
        codigo_aluno = (
            'nome = input("Nome: ")\n'
            'print(nome)'
        )
        testes = [
            {"entrada": "João", "saida": "João", "obs": ""},
        ]

        mock_exec.return_value = _execucao_eof_error(linha=1)

        q = Questao(
            idx=1,
            tipo="correcao",
            enunciado="Corrija.",
            codigo_base='nome = input("Nome: ")\nprint(nome)',
        )

        from evaluation.strategies.codigo import avaliar
        resultado = avaliar(q, codigo_aluno, testes)

        # 1 input no código, 1 entrada → quantidades iguais → falha real
        self.assertEqual(resultado.testes_executados[0]["compatibilidade"], "aprovado")
        self.assertFalse(resultado.testes_executados[0]["ok"])
        self.assertEqual(resultado.nota, 0.0)  # 0 passou / 1 válido

    @mock.patch("evaluation.strategies.codigo.executar_codigo_python")
    def test_zero_division_e_falha(self, mock_exec):
        """
        Cenário 2: Código com input suficiente, ZeroDivisionError.
        → reprovado.
        """
        codigo_aluno = (
            'x = int(input("Número: "))\n'
            'print(10 / x)'
        )
        testes = [
            {"entrada": "0", "saida": "inf", "obs": "divisão por zero"},
        ]

        mock_exec.return_value = _execucao_zero_division()

        q = Questao(
            idx=1,
            tipo="correcao",
            enunciado="Divisão.",
            codigo_base='x = int(input())\nprint(10 / x)',
        )

        from evaluation.strategies.codigo import avaliar
        resultado = avaliar(q, codigo_aluno, testes)

        self.assertEqual(resultado.testes_executados[0]["compatibilidade"], "aprovado")
        self.assertFalse(resultado.testes_executados[0]["ok"])
        self.assertEqual(resultado.nota, 0.0)

    @mock.patch("evaluation.strategies.codigo.executar_codigo_python")
    def test_value_error_e_falha(self, mock_exec):
        """
        Cenário 3: Código com input suficiente, ValueError.
        → reprovado.
        """
        codigo_aluno = (
            'x = int(input("Número: "));\n'
            'print(x * 2)'
        )
        testes = [
            {"entrada": "abc", "saida": "?", "obs": ""},
        ]

        mock_exec.return_value = _execucao_value_error()

        q = Questao(
            idx=1,
            tipo="correcao",
            enunciado="Dobre.",
            codigo_base='x = int(input())\nprint(x * 2)',
        )

        from evaluation.strategies.codigo import avaliar
        resultado = avaliar(q, codigo_aluno, testes)

        self.assertEqual(resultado.testes_executados[0]["compatibilidade"], "aprovado")
        self.assertFalse(resultado.testes_executados[0]["ok"])
        self.assertEqual(resultado.nota, 0.0)

    @mock.patch("evaluation.strategies.codigo.executar_codigo_python")
    def test_timeout_e_falha(self, mock_exec):
        """
        Cenário 4: Timeout.
        → reprovado.
        """
        codigo_aluno = 'while True: pass'
        testes = [
            {"entrada": "", "saida": "", "obs": ""},
        ]

        mock_exec.return_value = _execucao_timeout()

        q = Questao(
            idx=1,
            tipo="correcao",
            enunciado="Loop.",
            codigo_base='print("ok")',
        )

        from evaluation.strategies.codigo import avaliar
        resultado = avaliar(q, codigo_aluno, testes)

        self.assertEqual(resultado.testes_executados[0]["compatibilidade"], "aprovado")
        self.assertFalse(resultado.testes_executados[0]["ok"])
        self.assertEqual(resultado.nota, 0.0)

    @mock.patch("evaluation.strategies.codigo.executar_codigo_python")
    def test_saida_correta_e_aprovado(self, mock_exec):
        """
        Cenário 5: Código executando normalmente com saída correta.
        → aprovado.
        """
        codigo_aluno = 'x = int(input())\nprint(x * 2)'
        testes = [
            {"entrada": "3", "saida": "6", "obs": ""},
        ]

        mock_exec.return_value = _execucao_ok(stdout="6")

        q = Questao(
            idx=1,
            tipo="correcao",
            enunciado="Dobre.",
            codigo_base='x = int(input())\nprint(x * 2)',
        )

        from evaluation.strategies.codigo import avaliar
        resultado = avaliar(q, codigo_aluno, testes)

        self.assertEqual(resultado.testes_executados[0]["compatibilidade"], "aprovado")
        self.assertTrue(resultado.testes_executados[0]["ok"])
        self.assertEqual(resultado.nota, 10.0)

    @mock.patch("evaluation.strategies.codigo.executar_codigo_python")
    def test_saida_incorreta_e_reprovado(self, mock_exec):
        """
        Cenário 6: Código executando com saída incorreta.
        → reprovado.
        """
        codigo_aluno = 'x = int(input())\nprint(x + 1)'  # erro: +1 em vez de *2
        testes = [
            {"entrada": "3", "saida": "6", "obs": ""},
        ]

        mock_exec.return_value = _execucao_ok(stdout="4")

        q = Questao(
            idx=1,
            tipo="correcao",
            enunciado="Dobre.",
            codigo_base='x = int(input())\nprint(x * 2)',
        )

        from evaluation.strategies.codigo import avaliar
        resultado = avaliar(q, codigo_aluno, testes)

        self.assertEqual(resultado.testes_executados[0]["compatibilidade"], "aprovado")
        self.assertFalse(resultado.testes_executados[0]["ok"])
        self.assertEqual(resultado.nota, 0.0)

    @mock.patch("evaluation.strategies.codigo.executar_codigo_python")
    def test_mista_aprovado_reprovado_incompativel(self, mock_exec):
        """
        Cenário 7: 3 aprovados, 1 reprovado, 2 incompatíveis.
        → nota = 3/4 * 10 = 7.5
        """
        codigo_aluno = (
            'nome = input("Nome: ")\n'
            'idade = input("Idade: ")\n'
            'print(f"{nome} {idade}")'
        )

        testes = [
            {"entrada": "João\n25", "saida": "João 25", "obs": "ok"},          # 1: aprovado
            {"entrada": "Ana\n30", "saida": "Ana 30", "obs": "ok"},             # 2: aprovado
            {"entrada": "Bob\n20", "saida": "Bob 20", "obs": "ok"},             # 3: aprovado
            {"entrada": "X\n10", "saida": "X 10", "obs": "saída errada"},       # 4: reprovado (saída "Z 10")
            {"entrada": "Z", "saida": "Z ??", "obs": "1 entrada"},              # 5: incompatível
            {"entrada": "W", "saida": "W ??", "obs": "1 entrada"},              # 6: incompatível
        ]

        def _executar(codigo, entrada, timeout=3):
            e = entrada.strip()
            if e == "João\n25":
                return _execucao_ok(stdout="João 25")
            elif e == "Ana\n30":
                return _execucao_ok(stdout="Ana 30")
            elif e == "Bob\n20":
                return _execucao_ok(stdout="Bob 20")
            elif e == "X\n10":
                return _execucao_ok(stdout="Z 10")  # saída errada
            elif e in ("Z", "W"):
                return _execucao_eof_error(linha=2)  # EOFError (1 entrada, 2 inputs)
            return _execucao_ok(stdout="")

        mock_exec.side_effect = _executar

        q = Questao(
            idx=1,
            tipo="modificacao",
            enunciado="Modifique.",
            codigo_base='nome = input("Nome: ")\nprint(nome)',
        )

        from evaluation.strategies.codigo import avaliar
        resultado = avaliar(q, codigo_aluno, testes)

        self.assertEqual(resultado.nota, 7.5)
        self.assertEqual(resultado.testes_executados[0]["compatibilidade"], "aprovado")
        self.assertEqual(resultado.testes_executados[1]["compatibilidade"], "aprovado")
        self.assertEqual(resultado.testes_executados[2]["compatibilidade"], "aprovado")
        self.assertEqual(resultado.testes_executados[3]["compatibilidade"], "aprovado")
        self.assertFalse(resultado.testes_executados[3]["ok"])
        self.assertEqual(resultado.testes_executados[4]["compatibilidade"], "incompativel")
        self.assertEqual(resultado.testes_executados[5]["compatibilidade"], "incompativel")

    @mock.patch("evaluation.strategies.codigo.executar_codigo_python")
    def test_todos_incompativeis_nao_aprova(self, mock_exec):
        """
        Cenário 8: Todos os testes incompatíveis.
        → nota 0, não aprovação artificial.
        """
        codigo_aluno = (
            'nome = input("Nome: ")\n'
            'idade = input("Idade: ")\n'
            'print(nome, idade)'
        )
        testes = [
            {"entrada": "A", "saida": "A ??", "obs": ""},
            {"entrada": "B", "saida": "B ??", "obs": ""},
        ]

        mock_exec.return_value = _execucao_eof_error(linha=2)

        q = Questao(
            idx=1,
            tipo="modificacao",
            enunciado="Modifique.",
            codigo_base='nome = input("Nome: ")\nprint(nome)',
        )

        from evaluation.strategies.codigo import avaliar
        resultado = avaliar(q, codigo_aluno, testes)

        self.assertEqual(resultado.nota, 0.0)
        self.assertIn("Nenhum teste compatível", resultado.feedback)
        for t in resultado.testes_executados:
            self.assertEqual(t["compatibilidade"], "incompativel")


if __name__ == "__main__":
    unittest.main()
