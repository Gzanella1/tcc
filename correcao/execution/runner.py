#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
execution/runner.py

Execucao de codigo Python em container descartavel, sem rede e com limites.

Observacao importante: containerizacao reduz bastante o risco operacional,
mas nao torna a execucao "100% segura". A seguranca depende do runtime de
container, do kernel e da configuracao do ambiente hospedeiro.
"""

from __future__ import annotations

import ast
import os
import selectors
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from config import (
    CONTAINER_IMAGE,
    CONTAINER_RUNTIME,
    EXEC_CPU_SECONDS,
    EXEC_MAX_CODE_BYTES,
    EXEC_MAX_OPEN_FILES,
    EXEC_MAX_PROCESSES,
    EXEC_MAX_STDERR_BYTES,
    EXEC_MAX_STDIN_BYTES,
    EXEC_MAX_STDOUT_BYTES,
    EXEC_MEMORY_MB,
    EXEC_TIMEOUT_SECONDS,
)


@dataclass(frozen=True)
class ExecutionLimits:
    timeout_wall_seconds: int = EXEC_TIMEOUT_SECONDS
    cpu_seconds: int = EXEC_CPU_SECONDS
    memory_mb: int = EXEC_MEMORY_MB
    max_processes: int = EXEC_MAX_PROCESSES
    max_open_files: int = EXEC_MAX_OPEN_FILES
    max_code_bytes: int = EXEC_MAX_CODE_BYTES
    max_stdin_bytes: int = EXEC_MAX_STDIN_BYTES
    max_stdout_bytes: int = EXEC_MAX_STDOUT_BYTES
    max_stderr_bytes: int = EXEC_MAX_STDERR_BYTES


@dataclass
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    returncode: Optional[int] = None
    timeout: bool = False
    erro_execucao: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    limit_exceeded: bool = False
    backend: str = "container"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "timeout": self.timeout,
            "erro_execucao": self.erro_execucao,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "limit_exceeded": self.limit_exceeded,
            "backend": self.backend,
        }


def verificar_sintaxe_python(codigo: str) -> Tuple[bool, str]:
    """
    Verifica se o codigo Python e sintaticamente valido.
    Retorna (True, "") se valido, ou (False, mensagem_de_erro) se invalido.
    """
    try:
        ast.parse(codigo)
        return True, ""
    except SyntaxError as e:
        linha = f" linha {e.lineno}" if e.lineno else ""
        return False, f"{e.msg}{linha}"
    except Exception as e:
        return False, str(e)


def _bytes_len(texto: str) -> int:
    return len((texto or "").encode("utf-8", errors="replace"))


def _limite_excedido(mensagem: str) -> Dict[str, Any]:
    return ExecutionResult(
        erro_execucao=mensagem,
        limit_exceeded=True,
    ).to_dict()


def _resolver_runtime() -> Optional[str]:
    if CONTAINER_RUNTIME and CONTAINER_RUNTIME != "auto":
        return CONTAINER_RUNTIME if shutil.which(CONTAINER_RUNTIME) else None

    for candidato in ("podman", "docker"):
        if shutil.which(candidato):
            return candidato
    return None


def _montar_comando_container(
    runtime: str,
    nome: str,
    workdir: Path,
    limits: ExecutionLimits,
) -> list[str]:
    base = [
        runtime,
        "run",
        "--rm",
        "--name",
        nome,
        "--network",
        "none",
        "--user",
        "65534:65534",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=16m",
        "--memory",
        f"{limits.memory_mb}m",
        "--pids-limit",
        str(limits.max_processes),
        "--cpus",
        "1",
        "-i",
    ]

    base.extend([
        "--security-opt",
        "no-new-privileges",
        "--ulimit",
        f"cpu={limits.cpu_seconds}",
        "--ulimit",
        f"nofile={limits.max_open_files}:{limits.max_open_files}",
        "--ulimit",
        f"nproc={limits.max_processes}:{limits.max_processes}",
    ])

    base.extend([
        "-v",
        f"{workdir}:/work:ro",
        "-w",
        "/work",
        CONTAINER_IMAGE,
        "python",
        "-I",
        "/work/resposta_aluno.py",
    ])
    return base


def _append_limitado(buffer: bytearray, chunk: bytes, limite: int) -> bool:
    if len(buffer) >= limite:
        return True

    restante = limite - len(buffer)
    buffer.extend(chunk[:restante])
    return len(chunk) > restante


def _coletar_saida_limitada(
    proc: subprocess.Popen,
    limits: ExecutionLimits,
) -> tuple[bytes, bytes, bool, bool, bool]:
    """
    Le stdout/stderr ate que os streams fechem (EOF) ou o prazo externo expire.

    timed_out=True e retornado assim que o deadline de parede
    (limits.timeout_wall_seconds) e atingido, independentemente do estado
    do container. O fim da coleta sem timeout ocorre por EOF dos streams,
    sinal natural de que o processo externo terminou.
    """
    selector = selectors.DefaultSelector()
    stdout = bytearray()
    stderr = bytearray()
    stdout_truncated = False
    stderr_truncated = False
    timed_out = False
    deadline = time.monotonic() + limits.timeout_wall_seconds

    if proc.stdout is not None:
        selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
    if proc.stderr is not None:
        selector.register(proc.stderr, selectors.EVENT_READ, "stderr")

    try:
        while selector.get_map():
            restante = deadline - time.monotonic()
            if restante <= 0:
                timed_out = True
                break

            eventos = selector.select(timeout=min(0.1, restante))

            for key, _ in eventos:
                stream = key.fileobj
                try:
                    chunk = os.read(stream.fileno(), 4096)
                except OSError:
                    chunk = b""

                if not chunk:
                    selector.unregister(stream)
                    continue

                if key.data == "stdout":
                    stdout_truncated = (
                        _append_limitado(stdout, chunk, limits.max_stdout_bytes)
                        or stdout_truncated
                    )
                else:
                    stderr_truncated = (
                        _append_limitado(stderr, chunk, limits.max_stderr_bytes)
                        or stderr_truncated
                    )
    finally:
        selector.close()

    return bytes(stdout), bytes(stderr), stdout_truncated, stderr_truncated, timed_out


def _remover_container(runtime: str, nome: str) -> None:
    try:
        subprocess.run(
            [runtime, "rm", "-f", nome],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        pass


def _encerrar_processo(proc: Optional[subprocess.Popen]) -> None:
    """
    Encerra o processo externo (docker/podman run) e aguarda somente a
    finalizacao apos o encerramento explicito (nao e um segundo prazo de
    execucao).
    """
    if proc is None or proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

    try:
        proc.wait(timeout=2)
    except Exception:
        pass


def _fechar_pipes(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return

    for stream in (proc.stdout, proc.stderr):
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass


def _encerrar_execucao(runtime: str, nome: str, proc: Optional[subprocess.Popen]) -> None:
    """
    Encerramento por timeout, nesta ordem:
    1) mata o processo externo do runtime;
    2) garante a remocao do container.
    Idempotente: chamadas repetidas nao tem efeito adicional.
    """
    _encerrar_processo(proc)
    _remover_container(runtime, nome)


def executar_codigo_python(
    codigo: str,
    entrada: str,
    timeout: int = 3,
) -> Dict[str, Any]:
    """
    Executa codigo Python em container descartavel.

    A interface publica permanece compativel com o runner anterior:
    retorna um dicionario contendo stdout, stderr, returncode, timeout e
    erro_execucao. Campos novos informam truncamento, limite e backend.

    Distincao de desfechos:
    - timeout de parede: timeout=True, erro_execucao="Timeout", returncode=None;
    - limite de CPU (--ulimit cpu): processo interno morto pelo kernel,
      observado como returncode != 0 (runtime error), timeout permanece False;
    - truncamento de stdout/stderr: stdout_truncated/stderr_truncated e
      limit_exceeded=True, sem alterar timeout nem returncode;
    - erro de runtime: returncode != 0 com stderr preservado.
    """
    codigo = codigo or ""
    entrada = entrada if entrada is not None else ""
    limits = ExecutionLimits(
        timeout_wall_seconds=timeout,
        cpu_seconds=max(EXEC_CPU_SECONDS, timeout + 1),
    )

    if not codigo.strip():
        return ExecutionResult(erro_execucao="Código vazio").to_dict()

    if _bytes_len(codigo) > limits.max_code_bytes:
        return _limite_excedido("Limite de tamanho do código excedido")

    if _bytes_len(entrada) > limits.max_stdin_bytes:
        return _limite_excedido("Limite de tamanho da entrada excedido")

    runtime = _resolver_runtime()
    if not runtime:
        return ExecutionResult(
            erro_execucao="Runtime de container indisponível",
        ).to_dict()

    nome_container = f"correcao-{uuid.uuid4().hex}"

    with tempfile.TemporaryDirectory(prefix="correcao-run-") as td:
        workdir = Path(td)
        codigo_path = workdir / "resposta_aluno.py"
        entrada_path = workdir / "stdin.txt"
        codigo_path.write_text(codigo, encoding="utf-8")
        entrada_path.write_text(entrada, encoding="utf-8")

        # O container roda como 65534:65534 com montagem somente leitura;
        # sem estas permissoes o interpreter nao consegue ler /work.
        workdir.chmod(0o755)
        codigo_path.chmod(0o644)
        entrada_path.chmod(0o644)

        cmd = _montar_comando_container(runtime, nome_container, workdir, limits)
        proc: Optional[subprocess.Popen] = None

        try:
            with entrada_path.open("rb") as stdin_file:
                proc = subprocess.Popen(
                    cmd,
                    stdin=stdin_file,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )

                stdout_b, stderr_b, out_trunc, err_trunc, timed_out = (
                    _coletar_saida_limitada(proc, limits)
                )

                if timed_out:
                    # Prazo de parede expirado: o resultado nao depende do
                    # estado do container. Primeiro o processo externo,
                    # depois a remocao do container.
                    _encerrar_execucao(runtime, nome_container, proc)
                    returncode = None
                    _fechar_pipes(proc)
                else:
                    # EOF nos streams indica processo ja finalizado; wait()
                    # apenas recolhe o status final, sem novo prazo.
                    returncode = proc.wait()
                    _fechar_pipes(proc)

            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")

            erro_execucao = ""
            if timed_out:
                erro_execucao = "Timeout"
            elif returncode not in (0, None):
                erro_execucao = "Processo retornou código diferente de zero"

            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
                timeout=timed_out,
                erro_execucao=erro_execucao,
                stdout_truncated=out_trunc,
                stderr_truncated=err_trunc,
                limit_exceeded=out_trunc or err_trunc,
            ).to_dict()

        except FileNotFoundError:
            return ExecutionResult(
                erro_execucao="Falha ao iniciar container: runtime não encontrado",
            ).to_dict()
        except Exception as e:
            return ExecutionResult(
                erro_execucao=f"Falha ao iniciar container: {e}",
            ).to_dict()
        finally:
            # Limpeza idempotente, sem nova espera significativa: encerra
            # apenas processos que porventura tenham sobrevivido e garante
            # a remocao do container.
            if proc is not None and proc.poll() is None:
                _encerrar_processo(proc)
                _fechar_pipes(proc)
            _remover_container(runtime, nome_container)


def executar_codigo_python_sem_entrada(codigo: str, timeout: int = 3) -> Dict[str, Any]:
    """Atalho para executar codigo Python sem fornecer entrada via stdin."""
    return executar_codigo_python(codigo, "", timeout=timeout)
