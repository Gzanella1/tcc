#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
execution/runner.py

Execução isolada de código Python em subprocessos com timeout.
Inclui verificação de sintaxe via ast.parse.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

from utils.text import normalizar_texto


def verificar_sintaxe_python(codigo: str) -> Tuple[bool, str]:
    """
    Verifica se o código Python é sintaticamente válido.
    Retorna (True, "") se válido, ou (False, mensagem_de_erro) se inválido.
    """
    try:
        ast.parse(codigo)
        return True, ""
    except SyntaxError as e:
        linha = f" linha {e.lineno}" if e.lineno else ""
        return False, f"{e.msg}{linha}"
    except Exception as e:
        return False, str(e)


def executar_codigo_python(
    codigo: str,
    entrada: str,
    timeout: int = 3,
) -> Dict[str, Any]:
    """
    Executa código Python em processo separado usando o interpretador atual.
    O código é gravado em um arquivo temporário e executado com -I (modo isolado).

    Retorna um dicionário com:
        stdout       : saída padrão capturada
        stderr       : saída de erro capturada
        returncode   : código de retorno do processo (None se timeout/erro)
        timeout      : True se o processo excedeu o tempo limite
        erro_execucao: descrição do erro, se houver
    """
    codigo  = normalizar_texto(codigo)
    entrada = entrada if entrada is not None else ""

    resultado: Dict[str, Any] = {
        "stdout":        "",
        "stderr":        "",
        "returncode":    None,
        "timeout":       False,
        "erro_execucao": "",
    }

    if not codigo.strip():
        resultado["erro_execucao"] = "Código vazio"
        return resultado

    with tempfile.TemporaryDirectory() as td:
        caminho = Path(td) / "resposta_aluno.py"
        caminho.write_text(codigo, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(caminho)],
                input=entrada,
                text=True,
                capture_output=True,
                timeout=timeout,
                cwd=td,
            )
            resultado["stdout"]     = proc.stdout or ""
            resultado["stderr"]     = proc.stderr or ""
            resultado["returncode"] = proc.returncode
        except subprocess.TimeoutExpired as e:
            resultado["timeout"]       = True
            resultado["stdout"]        = e.stdout or ""
            resultado["stderr"]        = e.stderr or ""
            resultado["erro_execucao"] = "Timeout"
        except Exception as e:
            resultado["erro_execucao"] = str(e)

    return resultado


def executar_codigo_python_sem_entrada(codigo: str, timeout: int = 3) -> Dict[str, Any]:
    """Atalho para executar código Python sem fornecer entrada via stdin."""
    return executar_codigo_python(codigo, "", timeout=timeout)
