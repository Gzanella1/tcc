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
# a entrada gerada e exige: execução sem timeout, exit code 0, saída não
# vazia e CONSUMO INTEGRAL do stdin (nenhuma linha não vazia pode sobrar
# sem ser lida — sobra indica que o formato da entrada não casa com a forma
# como o código lê, ex.: "4\n9\n2\n7" para `input().split()`, que lê só "4").
# Questões cuja entrada não valida são rejeitadas na GERAÇÃO.
#
# Formatos alternativos (fallback):
#   A serialização padrão acima é SEMPRE tentada primeiro; se ela valida,
#   o resultado é idêntico ao comportamento anterior. Só quando ela falha
#   (exceto por timeout) tenta-se variantes em que cada lista de escalares
#   vira UMA linha com os itens unidos por um separador — primeiro os
#   separadores literais usados em `.split("<sep>")` no código-base, depois
#   espaço e vírgula. A primeira variante que valida é a escolhida.

from __future__ import annotations

import json
import re
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


# ─── Execução de referência com verificação de consumo do stdin ──────────────

# Wrapper executado no subprocesso: roda o código-base como __main__ e, ao
# final (término normal ou sys.exit(0)), lê o que SOBROU no stdin e grava
# num arquivo à parte — sem tocar no stdout do código-base. Se o código
# falhar, nada é gravado (a falha já rejeita a entrada por outro motivo).
_WRAPPER_CONSUMO = r'''
import builtins, sys
_codigo_path, _resto_path = sys.argv[1], sys.argv[2]
sys.argv = [_codigo_path]

def _registrar_resto():
    try:
        resto = sys.stdin.read()
    except Exception:
        return
    with open(_resto_path, "w", encoding="utf-8") as f:
        f.write(resto)

with open(_codigo_path, encoding="utf-8") as f:
    _fonte = f.read()
_ns = {"__name__": "__main__", "__file__": _codigo_path, "__builtins__": builtins}
try:
    exec(compile(_fonte, _codigo_path, "exec"), _ns)
except SystemExit as e:
    if e.code in (None, 0):
        _registrar_resto()
    raise
_registrar_resto()
'''


def executar_referencia(codigo: str, entrada: str = "", timeout: int = 5) -> dict:
    """
    Executa o código-base em subprocesso ISOLADO (python -I) com a entrada
    fornecida. Retorna dict com returncode/stdout/stderr/timeout e
    stdin_restante (o que o código NÃO leu do stdin; None quando não foi
    possível determinar, ex.: execução com erro).

    Usado apenas na VALIDAÇÃO DA GERAÇÃO — não acopla esta pipeline ao
    runner de correção.
    """
    codigo = codigo or ""
    entrada = entrada or ""
    with tempfile.TemporaryDirectory(prefix="previsao-referencia-") as tmp:
        dirpath = Path(tmp)
        codigo_path = dirpath / "codigo_base.py"
        wrapper_path = dirpath / "_wrapper_consumo.py"
        resto_path = dirpath / "stdin_restante.txt"
        stdin_path = dirpath / "stdin.txt"
        codigo_path.write_text(codigo, encoding="utf-8")
        wrapper_path.write_text(_WRAPPER_CONSUMO, encoding="utf-8")
        stdin_path.write_text(entrada, encoding="utf-8")
        try:
            with stdin_path.open("rb") as f:
                proc = subprocess.run(
                    [sys.executable, "-I", str(wrapper_path),
                     str(codigo_path), str(resto_path)],
                    stdin=f,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                )
        except subprocess.TimeoutExpired:
            return {"returncode": None, "stdout": "", "stderr": "",
                    "timeout": True, "stdin_restante": None}
        restante = (
            resto_path.read_text(encoding="utf-8", errors="replace")
            if resto_path.exists() else None
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout.decode("utf-8", errors="replace"),
            "stderr": proc.stderr.decode("utf-8", errors="replace"),
            "timeout": False,
            "stdin_restante": restante,
        }


# ─── Variantes de formato (fallback) ─────────────────────────────────────────

def _eh_escalar(valor: Any) -> bool:
    return not isinstance(valor, (list, dict))


def _compactar_listas(dados: Any, sep: str, topo: bool = True) -> Any:
    """
    Substitui cada lista composta só de escalares por UMA string com os
    itens unidos por `sep`. A estrutura restante (dicts, listas de listas)
    é preservada para que os serializadores do registro continuem valendo.
    Ex. (sep=" "): [4, 9, 2, 7] → ["4 9 2 7"];  [[1, 2], [3, 4]] → ["1 2", "3 4"].
    """
    if isinstance(dados, list):
        if dados and all(_eh_escalar(x) for x in dados):
            unida = sep.join(str(x) for x in dados)
            return [unida] if topo else unida
        return [_compactar_listas(x, sep, topo=False) for x in dados]
    if isinstance(dados, dict):
        return {k: _compactar_listas(v, sep, topo=False) for k, v in dados.items()}
    return dados


def _separadores_candidatos(codigo_base: str) -> List[str]:
    """Separadores de `.split("<sep>")` do código, seguidos de espaço e vírgula."""
    literais = re.findall(r"""\.split\(\s*(['"])(.+?)\1\s*\)""", codigo_base or "")
    seps: List[str] = []
    for sep in [s for _q, s in literais] + [" ", ","]:
        sep = sep.encode("utf-8").decode("unicode_escape") if "\\" in sep else sep
        if sep and sep not in seps and "\n" not in sep:
            seps.append(sep)
    return seps


def entradas_alternativas(dados: Any, codigo_base: str) -> List[str]:
    """
    Serializações alternativas de dados_entrada (sem a padrão), em ordem de
    preferência e sem repetições. Vazia quando os dados não têm listas de
    escalares (nada a compactar).
    """
    try:
        padrao = serializar_entrada(dados)
    except FormatoEntradaDesconhecido:
        return []
    vistas = {padrao}
    alternativas: List[str] = []
    for sep in _separadores_candidatos(codigo_base):
        try:
            entrada = serializar_entrada(_compactar_listas(dados, sep))
        except FormatoEntradaDesconhecido:
            continue
        if entrada.strip() and entrada not in vistas:
            vistas.add(entrada)
            alternativas.append(entrada)
    return alternativas


# ─── Validação ────────────────────────────────────────────────────────────────

def _validar_execucao(codigo_base: str, entrada: str, timeout: int) -> tuple:
    """
    Executa o código-base com `entrada` e retorna (ok, motivo, saida, timeout).
    """
    execucao = executar_referencia(codigo_base, entrada, timeout=timeout)
    if execucao["timeout"]:
        return False, "timeout ao executar o código-base com a entrada gerada", "", True
    if execucao["returncode"] != 0:
        return (
            False,
            f"código-base não executou com a entrada gerada "
            f"(returncode={execucao['returncode']})",
            "",
            False,
        )
    if not (execucao["stdout"] or "").strip():
        return False, "execução de referência não produziu saída", "", False
    restante = execucao.get("stdin_restante")
    if restante is not None and restante.strip():
        return (
            False,
            f"entrada gerada não foi totalmente lida pelo código-base "
            f"(sobrou: {restante.strip()[:60]!r}) — formato incompatível",
            "",
            False,
        )
    return True, "", execucao["stdout"], False


def gerar_entrada_testes(dados: Any, codigo_base: str, timeout: int = 5) -> tuple:
    """
    Serializa dados_entrada e valida a questão por execução de referência.

    Tenta primeiro a serialização PADRÃO; se ela valida, o resultado é o
    mesmo de sempre. Se falhar (exceto timeout), tenta as variantes de
    entradas_alternativas() e usa a primeira que validar.

    Retorna (ok, motivo, entrada, saida):
      ok      – True apenas se a entrada serializa, o código executa sem
                timeout/erro, produz saída não vazia e lê todo o stdin;
      motivo  – descrição do motivo da rejeição quando ok=False (sempre o
                motivo da serialização padrão);
      entrada – a string entradaTestes escolhida (vazia quando rejeitada);
      saida   – stdout cru da execução de referência.
    """
    try:
        entrada = serializar_entrada(dados)
    except FormatoEntradaDesconhecido as exc:
        return False, str(exc), "", ""

    if not (entrada or "").strip():
        return False, "dados_entrada não produziu entrada (entradaTestes vazio)", "", ""

    ok, motivo, saida, estourou = _validar_execucao(codigo_base, entrada, timeout)
    if ok:
        return True, "", entrada, saida
    if estourou:
        return False, motivo, "", ""

    for alternativa in entradas_alternativas(dados, codigo_base):
        ok_alt, _motivo_alt, saida_alt, _ = _validar_execucao(
            codigo_base, alternativa, timeout
        )
        if ok_alt:
            return True, "", alternativa, saida_alt

    return False, motivo, "", ""
