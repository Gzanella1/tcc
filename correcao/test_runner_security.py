#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from config import CONTAINER_IMAGE
from execution import runner


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


class RunnerUnitTests(unittest.TestCase):
    def test_sintaxe_valida(self):
        ok, erro = runner.verificar_sintaxe_python("print('ok')")
        self.assertTrue(ok)
        self.assertEqual(erro, "")

    def test_sintaxe_invalida(self):
        ok, erro = runner.verificar_sintaxe_python("if True print('x')")
        self.assertFalse(ok)
        self.assertIn("invalid syntax", erro)

    @mock.patch("execution.runner._resolver_runtime", return_value=None)
    def test_container_indisponivel(self, _):
        resultado = runner.executar_codigo_python("print('x')", "")
        self.assertEqual(resultado["backend"], "container")
        self.assertIn("Runtime de container indisponível", resultado["erro_execucao"])

    def test_codigo_grande_nao_executa(self):
        codigo = "x = 1\n" * 20000
        resultado = runner.executar_codigo_python(codigo, "")
        self.assertTrue(resultado["limit_exceeded"])
        self.assertIn("código", resultado["erro_execucao"])

    def test_entrada_grande_nao_executa(self):
        entrada = "x" * 70000
        resultado = runner.executar_codigo_python("print('ok')", entrada)
        self.assertTrue(resultado["limit_exceeded"])
        self.assertIn("entrada", resultado["erro_execucao"])

    def test_comando_container_nao_monta_projeto_e_remove_rede(self):
        cmd = runner._montar_comando_container(
            runtime="docker",
            nome="teste",
            workdir=runner.Path("/tmp/correcao"),
            limits=runner.ExecutionLimits(),
        )
        self.assertIn("--network", cmd)
        self.assertIn("none", cmd)
        self.assertIn("--read-only", cmd)
        self.assertIn("--user", cmd)
        self.assertIn("65534:65534", cmd)
        self.assertIn("/tmp/correcao:/work:ro", cmd)


@unittest.skipUnless(
    RUNTIME,
    f"Docker/Podman indisponível ou imagem {CONTAINER_IMAGE!r} ausente localmente",
)
class RunnerContainerIntegrationTests(unittest.TestCase):
    def _executar(self, codigo: str, entrada: str = "", timeout: int = 3):
        return runner.executar_codigo_python(codigo, entrada, timeout=timeout)

    def test_codigo_normal(self):
        resultado = self._executar("nome = input()\nprint('Ola', nome)", "Ana\n")
        self.assertEqual(resultado["returncode"], 0)
        self.assertEqual(resultado["stdout"].strip(), "Ola Ana")
        self.assertFalse(resultado["timeout"])

    def test_codigo_com_erro_runtime(self):
        resultado = self._executar("raise ValueError('falhou')")
        self.assertNotEqual(resultado["returncode"], 0)
        self.assertIn("ValueError", resultado["stderr"])
        self.assertIn("código diferente de zero", resultado["erro_execucao"])

    def test_returncode_diferente_de_zero(self):
        resultado = self._executar("import sys\nsys.exit(7)")
        self.assertEqual(resultado["returncode"], 7)
        self.assertIn("código diferente de zero", resultado["erro_execucao"])

    def test_timeout_destroi_container(self):
        resultado = self._executar("while True:\n    pass", timeout=1)
        self.assertTrue(resultado["timeout"])
        self.assertEqual(resultado["erro_execucao"], "Timeout")
        self.assertFalse(self._containers_residuais())

    def test_stdout_grande_e_truncado(self):
        resultado = self._executar("while True:\n    print('x' * 1000)")
        self.assertTrue(resultado["limit_exceeded"])
        self.assertTrue(resultado["stdout_truncated"])
        self.assertLessEqual(len(resultado["stdout"].encode("utf-8")), 65536)

    def test_stderr_grande_e_truncado(self):
        codigo = "import sys\nwhile True:\n    print('e' * 1000, file=sys.stderr)"
        resultado = self._executar(codigo)
        self.assertTrue(resultado["limit_exceeded"])
        self.assertTrue(resultado["stderr_truncated"])
        self.assertLessEqual(len(resultado["stderr"].encode("utf-8")), 65536)

    def test_criacao_de_muitos_processos_e_limitada(self):
        codigo = (
            "import subprocess, time\n"
            "procs=[]\n"
            "for _ in range(200):\n"
            "    try:\n"
            "        procs.append(subprocess.Popen(['python', '-c', 'import time; time.sleep(5)']))\n"
            "    except Exception as e:\n"
            "        print(type(e).__name__)\n"
            "        break\n"
        )
        resultado = self._executar(codigo, timeout=2)
        self.assertTrue(resultado["timeout"] or resultado["returncode"] in (0, 1))
        self.assertFalse(self._containers_residuais())

    def test_tentativa_de_acesso_rede_falha(self):
        codigo = (
            "import socket\n"
            "s=socket.socket()\n"
            "s.settimeout(1)\n"
            "try:\n"
            "    s.connect(('1.1.1.1', 80))\n"
            "    print('rede-ok')\n"
            "except Exception as e:\n"
            "    print(type(e).__name__)\n"
        )
        resultado = self._executar(codigo, timeout=3)
        self.assertNotIn("rede-ok", resultado["stdout"])

    def test_tentativa_de_acessar_arquivo_do_host_falha(self):
        codigo = (
            "from pathlib import Path\n"
            "p=Path('/home/giovani/Músicas/tcc/conteudo/conhecimento.txt')\n"
            "print('existe' if p.exists() else 'nao-existe')\n"
        )
        resultado = self._executar(codigo)
        self.assertEqual(resultado["stdout"].strip(), "nao-existe")

    def test_limpeza_apos_erro(self):
        self._executar("raise RuntimeError('x')")
        self.assertFalse(self._containers_residuais())

    def _containers_residuais(self) -> bool:
        proc = subprocess.run(
            [RUNTIME, "ps", "-a", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        nomes = [linha.strip() for linha in proc.stdout.splitlines()]
        return any(nome.startswith("correcao-") for nome in nomes)


if __name__ == "__main__":
    unittest.main()
