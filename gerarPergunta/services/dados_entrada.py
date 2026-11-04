# =============================================================
# services/dados_entrada.py
# Transformação determinística dados_entrada → entradaTestes
# e validação da entrada por execução de referência.
# =============================================================
#
# Contrato de GERAÇÃO (PREVISAO):
#   - O LLM retorna "dados_entrada" como dado SEMÂNTICO estruturado
#     (dict/list), NUNCA como string de stdin pronta.
#   - Este módulo converte dados_entrada na string de entrada de
#     execução (entradaTestes) de forma 100% determinística, sem
#     depender do LLM nem de reparse de enunciado.
#   - A conversão é extensível: novos serializadores bastam implementar
#     aceita()/serializar() e ser registrados no registry.
#
# Convenções de serialização (ordem de tentativa):
#   1. lista linear            → uma linha por elemento
#   2. mapa com 1 dict aninhado→ linhas chave/valor do dict + linha em
#                                branco (sentinel que encerra laços de
#                                leitura "while True") + escalares restantes
#   3. dict de escalares       → uma linha por valor
#
# A validação de geração executa o código-base em subprocesso isolado com
# a entrada gerada e exige: execução sem timeout, exit code 0 e saída não
# vazia. Questões cuja entrada não valida são rejeitadas na GERAÇÃO.

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, List


class FormatoEntradaDesconhecido(ValueError):
    """dados_entrada não corresponde a nenhum serializador conhecido."""


class _SerializadorLinear:
    """Lista de valores → uma linha por elemento."""

    nome = "lista_linear"

    @staticmethod
    def aceita(dados: Any) -> bool:
        return isinstance(dados, list)

    @staticmethod
    def serializar(dados: List[Any]) -> str:
        linhas = []
        for item in dados:
            if isinstance(item, (list, dict)):
                # Aninhamento: serializa recursivamente e apara a quebra final.
                linhas.append(serializar_entrada(item).rstrip("\n"))
            else:
                linhas.append(str(item))
        return "\n".join(linhas) + "\n"


class _SerializadorMapaComBusca:
    """
    Mapa com exatamente um dict aninhado (o dado principal) e campos
    escalares restantes. Ex.: {"dicionario": {"a":5, ...}, "valor_busca": "5"}.

    Sequência de stdin:
      1. para cada par do dict aninhado:   <chave>\n<valor>\n
      2. uma linha EM BRANCO              (encerra laços "while True:")
      3. para cada escalar restante:       <valor>\n
    """

    nome = "mapa_pares_com_busca"

    @staticmethod
    def aceita(dados: Any) -> bool:
        if not isinstance(dados, dict) or not dados:
            return False
        return sum(1 for v in dados.values() if isinstance(v, dict)) == 1

    @staticmethod
    def serializar(dados: dict) -> str:
        linhas: List[str] = []
        finais: List[Any] = []
        for valor in dados.values():
            if isinstance(valor, dict):
                for chave, item in valor.items():
                    linhas.append(str(chave))
                    linhas.append(str(item))
            else:
                finais.append(valor)
        linhas.append("")                  # sentinela: fim da coleta de pares
        linhas.extend(str(v) for v in finais)
        return "\n".join(linhas) + "\n"


class _SerializadorDictEscalares:
    """Mapa composto apenas por escalares → uma linha por valor."""

    nome = "dict_escalares"

    @staticmethod
    def aceita(dados: Any) -> bool:
        return isinstance(dados, dict)

    @staticmethod
    def serializar(dados: dict) -> str:
        if not dados:
            return ""
        linhas = []
        for valor in dados.values():
            if isinstance(valor, (list, dict)):
                linhas.append(serializar_entrada(valor).rstrip("\n"))
            else:
                linhas.append(str(valor))
        return "\n".join(linhas) + "\n"


_REGISTRO: List[Any] = [
    _SerializadorLinear(),
    _SerializadorMapaComBusca(),
    _SerializadorDictEscalares(),
]


def registrar_serializador(serializador: Any) -> None:
    """Insere um novo serializador no topo da precedência de tentativa."""
    _REGISTRO.insert(0, serializador)


def serializar_entrada(dados: Any) -> str:
    """
    Converte dados_entrada estruturados na string de entrada de execução.

    Levanta FormatoEntradaDesconhecido quando o formato não corresponde a
    nenhum serializador conhecido.
    """
    if dados is None:
        raise FormatoEntradaDesconhecido("dados_entrada ausente")
    for serializador in _REGISTRO:
        if serializador.aceita(dados):
            return serializador.serializar(dados)
    raise FormatoEntradaDesconhecido(
        f"formato de dados_entrada não suportado: "
        f"{json.dumps(dados, ensure_ascii=False)[:120]}"
    )


def executar_referencia(codigo: str, entrada: str = "", timeout: int = 5) -> dict:
    """
    Executa o código-base em subprocesso ISOLADO (python -I) com a entrada
    fornecida. Retorna dict com returncode/stdout/stderr/timeout.

    Usado apenas na VALIDAÇÃO DA GERAÇÃO — não acopla esta pipeline ao
    runner de correção.
    """
    codigo = codigo or ""
    entrada = entrada or ""
    with tempfile.TemporaryDirectory(prefix="previsao-referencia-") as tmp:
        dirpath = Path(tmp)
        codigo_path = dirpath / "codigo_base.py"
        stdin_path = dirpath / "stdin.txt"
        codigo_path.write_text(codigo, encoding="utf-8")
        stdin_path.write_text(entrada, encoding="utf-8")
        try:
            with stdin_path.open("rb") as f:
                proc = subprocess.run(
                    [sys.executable, "-I", str(codigo_path)],
                    stdin=f,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                )
            return {
                "returncode": proc.returncode,
                "stdout": proc.stdout.decode("utf-8", errors="replace"),
                "stderr": proc.stderr.decode("utf-8", errors="replace"),
                "timeout": False,
            }
        except subprocess.TimeoutExpired:
            return {"returncode": None, "stdout": "", "stderr": "", "timeout": True}


def gerar_entrada_testes(dados: Any, codigo_base: str, timeout: int = 5) -> tuple:
    """
    Serializa dados_entrada e valida a questão por execução de referência.

    Retorna (ok, motivo, entrada, saida):
      ok      – True apenas se a entrada serializa, o código executa sem
                timeout/erro e produz saída não vazia;
      motivo  – descrição do motivo da rejeição quando ok=False;
      entrada – a string entradaTestes gerada (vazia quando rejeitada);
      saida   – stdout cru da execução de referência.
    """
    try:
        entrada = serializar_entrada(dados)
    except FormatoEntradaDesconhecido as exc:
        return False, str(exc), "", ""

    if not (entrada or "").strip():
        return False, "dados_entrada não produziu entrada (entradaTestes vazio)", "", ""

    execucao = executar_referencia(codigo_base, entrada, timeout=timeout)
    if execucao["timeout"]:
        return False, "timeout ao executar o código-base com a entrada gerada", "", ""
    if execucao["returncode"] != 0:
        return (
            False,
            f"código-base não executou com a entrada gerada "
            f"(returncode={execucao['returncode']})",
            "",
            "",
        )
    saida = (execucao["stdout"] or "").strip()
    if not saida:
        return False, "execução de referência não produziu saída", "", ""

    return True, "", entrada, execucao["stdout"]