#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils/text.py

Funções utilitárias para normalização, comparação e tokenização de texto.
"""

from __future__ import annotations

import ast
import difflib
import json
import re
import unicodedata
from typing import Dict, Iterable, List


def sem_acentos(texto: str) -> str:
    """Remove acentos e diacríticos de uma string."""
    texto = unicodedata.normalize("NFD", texto)
    return "".join(ch for ch in texto if unicodedata.category(ch) != "Mn")


def normalizar_label(label: str) -> str:
    """Normaliza um rótulo removendo acentos, espaços extras e caracteres especiais."""
    label = sem_acentos(label).lower().strip()
    label = re.sub(r"[^a-z0-9]+", " ", label)
    return re.sub(r"\s+", " ", label).strip()


def normalizar_texto(texto: str) -> str:
    """Normaliza quebras de linha e remove espaços em branco desnecessários."""
    if texto is None:
        return ""
    texto = str(texto).replace("\r\n", "\n").replace("\r", "\n")
    linhas = [ln.rstrip() for ln in texto.split("\n")]
    while linhas and not linhas[0].strip():
        linhas.pop(0)
    while linhas and not linhas[-1].strip():
        linhas.pop()
    return "\n".join(linhas).strip()


def compactar_texto(texto: str) -> str:
    """Remove quebras de linha e espaços extras, retornando texto em linha única."""
    texto = normalizar_texto(texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def tokenizar(texto: str) -> List[str]:
    """Divide o texto em tokens alfanuméricos normalizados."""
    texto = sem_acentos((texto or "").lower())
    return re.findall(r"[a-z0-9_]+", texto)


def comparar_textos(a: str, b: str) -> float:
    """
    Retorna uma similaridade entre 0 e 1 entre dois textos.
    Considera comparação exata, sem espaços e SequenceMatcher.
    """
    a0 = compactar_texto(a)
    b0 = compactar_texto(b)

    if a0 == b0:
        return 1.0

    a1 = a0.replace(" ", "")
    b1 = b0.replace(" ", "")
    if a1 == b1 and a1:
        return 0.98

    ratio = difflib.SequenceMatcher(None, a0.lower(), b0.lower()).ratio()
    return ratio


def saida_contem_esperado(saida_obtida: str, saida_esperada: str) -> bool:
    """
    Verifica se a saída obtida contém a saída esperada, preservando a ordem.

    Útil para avaliação de código/modificação: um print extra não deve
    invalidar a resposta se todos os valores esperados aparecerem corretamente.
    """
    esperada = normalizar_texto(saida_esperada)
    obtida = normalizar_texto(saida_obtida)

    if not esperada:
        return not obtida

    if esperada.lower() in obtida.lower():
        return True

    linhas_esperadas = [ln.strip().lower() for ln in esperada.splitlines() if ln.strip()]
    linhas_obtidas = [ln.strip().lower() for ln in obtida.splitlines() if ln.strip()]

    if not linhas_esperadas:
        return True

    pos = 0
    for linha_obtida in linhas_obtidas:
        if linhas_esperadas[pos] in linha_obtida:
            pos += 1
            if pos == len(linhas_esperadas):
                return True

    return False


def extrair_codigo(texto: str) -> str:
    """
    Extrai código entre cercas markdown.
    Se não houver cercas, retorna o texto bruto.
    """
    if not texto:
        return ""

    blocos = re.findall(
        r"```(?:python|py|c|cpp|java|javascript|txt|text)?\s*\n(.*?)```",
        texto,
        flags=re.S | re.I,
    )
    if blocos:
        return "\n\n".join(bloco.strip() for bloco in blocos if bloco.strip())

    return texto.strip()


def extrair_json(texto: str):
    """
    Tenta extrair e parsear um objeto JSON de um texto livre.
    Remove cercas markdown antes de tentar o parse.
    """
    if not texto:
        return None

    texto = texto.strip()
    texto = re.sub(r"^```[a-zA-Z]*", "", texto)
    texto = re.sub(r"```$", "", texto).strip()

    try:
        return json.loads(texto)
    except Exception:
        pass

    match = re.search(r"\{.*\}", texto, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return None


def exige_saida_no_enunciado(texto: str) -> bool:
    """
    Detecta se o enunciado realmente pede saída na tela / retorno.
    Evita penalizar questões que só pedem modificação estrutural do código.
    """
    t = sem_acentos((texto or "").lower())
    padroes = [
        r"\bprint(?:e|ar|em|ar)?\b",
        r"\bmostre\b",
        r"\bimprima\b",
        r"\bexiba\b",
        r"\bsaida\b",
        r"\bretorne\b",
        r"\breturn\b",
        r"\bdeve retornar\b",
        r"\bmostrar o resultado\b",
    ]
    return any(re.search(p, t) for p in padroes)


def codigo_tem_input(codigo: str) -> bool:
    """Verifica se um trecho de código Python usa a função input()."""
    if not codigo:
        return False
    return bool(re.search(r"\binput\s*\(", codigo))


# ─── Classificação de erros de execução ──────────────────────────────────────

# Classificações de erro usadas pela camada de avaliação para distinguir
# testes incompatíveis de erros reais do código do aluno.
ERRO_EOF_INCOMPATIVEL = "eof_incompativel"
ERRO_EOF_REAL = "eof_logico"
ERRO_RUNTIME = "erro_runtime"
ERRO_TIMEOUT = "timeout"
ERRO_NENHUM = ""


def classificar_erro_execucao(stderr: str, returncode) -> str:
    """
    Classifica o tipo de erro de execução a partir do traceback.

    Retorna uma das constantes:
        ERRO_EOF_INCOMPATIVEL  – EOFError e o contexto sugere incompatibilidade
                                 de interface (chamada adicional de input());
                                 caller deve confirmar com contagem de inputs.
        ERRO_EOF_REAL          – EOFError sem evidência de incompatibilidade
                                 (ex.: input() dentro de try/except, ou
                                 mesma quantidade de inputs que a entrada).
        ERRO_RUNTIME           – qualquer outra exceção (ZeroDivisionError,
                                 ValueError, TypeError, IndexError, etc.)
        ERRO_TIMEOUT           – (tratado separadamente; não chega aqui)
        ERRO_NENHUM            – sem erro detectado

    Limitação: EOFError nãoquantitative sozinho prova incompatibilidade.
    O caller DEVE combinar com contar_inputs_codigo() para decidir.
    Esta função仅 identifica que o erro É um EOFError (potencialmente
    incompatível) vs. outro tipo de erro real.
    """
    if returncode is None:
        return ERRO_NENHUM

    _stderr = (stderr or "").strip()
    _returncode = int(returncode or 0)

    if not _stderr and _returncode == 0:
        return ERRO_NENHUM

    # Detectar EOFError no traceback
    if "EOFError" in _stderr:
        # EOFError sem mais contexto → caller decide com contagem de inputs
        return ERRO_EOF_INCOMPATIVEL

    # Outros erros de execução
    if _returncode != 0 or "Traceback" in _stderr or "Error" in _stderr:
        return ERRO_RUNTIME

    return ERRO_NENHUM


def contar_inputs_codigo(codigo: str) -> int:
    """
    Conta chamadas estáticas a input() no código Python via AST.

    Retorna o número de nós Call que chamam input(). Conta todas as
    chamadas, incluindo condicionais e loops — é uma heurística
    conservadora.

    Limitações conhecidas:
    - inputs dentro de loops (for/while) são contados uma vez, mas
      podem ser executados N vezes.
    - inputs dinâmicos (ex.: getattr(sys, 'in'+'put')()) não são detectados.
    - Se o código tiver erro de sintaxe, retorna 0 (o caller já
      verificou sintaxe antes de chamar esta função).
    """
    if not codigo:
        return 0

    try:
        arvore = ast.parse(codigo)
    except SyntaxError:
        return 0

    count = 0
    for node in ast.walk(arvore):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "input":
            count += 1
    return count


def blocos_na_entrada(entrada: str) -> int:
    """
    Estima a quantidade de blocos de entrada fornecidos ao código.

    Cada bloco corresponde a uma chamada input() que será satisfeita.
    Usa como heurística a separação por quebras de linha simples.

    Exemplos:
        ""           → 0
        "João"       → 1
        "João\\n14:00"  → 2
        "a\\nb\\nc"    → 3
    """
    if not entrada:
        return 0
    linhas = entrada.split("\n")
    # Filtra linhas vazias no final (trailing newline comum)
    while linhas and not linhas[-1].strip():
        linhas.pop()
    return len(linhas)


def _normalizar_prompt(prompt: str) -> str:
    """Gera variantes razoáveis do prompt para remoção segura."""
    prompt = normalizar_texto(prompt)
    if not prompt:
        return ""
    return prompt


def _unique(seq: Iterable[str]) -> List[str]:
    vistos = set()
    saida: List[str] = []
    for item in seq:
        if not item:
            continue
        if item in vistos:
            continue
        vistos.add(item)
        saida.append(item)
    return saida


def _eh_chamada_input(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "input"
    )


def _prompt_literal_input(node: ast.Call) -> str:
    if not node.args:
        return ""

    arg0 = node.args[0]

    if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
        return _normalizar_prompt(arg0.value)

    # Casos simples de f-string literal, sem expressões interpoladas.
    if isinstance(arg0, ast.JoinedStr):
        partes: List[str] = []
        for parte in arg0.values:
            if not isinstance(parte, ast.Constant) or not isinstance(parte.value, str):
                return ""
            partes.append(parte.value)
        return _normalizar_prompt("".join(partes))

    return ""


def _nomes_alvo_atribuicao(target: ast.AST) -> List[str]:
    if isinstance(target, ast.Name):
        return [target.id]

    if isinstance(target, (ast.Tuple, ast.List)):
        nomes: List[str] = []
        for elemento in target.elts:
            nomes.extend(_nomes_alvo_atribuicao(elemento))
        return nomes

    if isinstance(target, ast.Attribute):
        try:
            return [ast.unparse(target)]
        except Exception:
            return [target.attr]

    return []


def _variavel_associada_input(pilha: List[ast.AST]) -> str:
    for node in reversed(pilha):
        if isinstance(node, ast.Assign):
            nomes: List[str] = []
            for target in node.targets:
                nomes.extend(_nomes_alvo_atribuicao(target))
            return ", ".join(nomes)

        if isinstance(node, ast.AnnAssign):
            return ", ".join(_nomes_alvo_atribuicao(node.target))

        if isinstance(node, ast.NamedExpr):
            return ", ".join(_nomes_alvo_atribuicao(node.target))

    return ""


def _conversor_input(pilha: List[ast.AST]) -> str:
    if not pilha:
        return ""

    pai = pilha[-1]
    if not isinstance(pai, ast.Call):
        return ""

    func = pai.func
    if isinstance(func, ast.Name) and func.id in {"int", "float", "str", "bool"}:
        return func.id

    return ""


class _AssinaturaInputsVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.pilha: List[ast.AST] = []
        self.assinatura: List[Dict[str, str]] = []

    def visit(self, node: ast.AST):  # type: ignore[override]
        self.pilha.append(node)
        try:
            return super().visit(node)
        finally:
            self.pilha.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if _eh_chamada_input(node):
            ancestrais = self.pilha[:-1]
            self.assinatura.append({
                "indice": str(len(self.assinatura) + 1),
                "variavel": _variavel_associada_input(ancestrais),
                "prompt": _prompt_literal_input(node),
                "conversor": _conversor_input(ancestrais),
            })

        self.generic_visit(node)


def _extrair_assinatura_inputs_fallback(codigo: str) -> List[Dict[str, str]]:
    """
    Fallback textual para códigos parcialmente inválidos.

    Não tenta interpretar Python completo; apenas preserva a ordem textual
    de chamadas input(...) e captura padrões simples de atribuição/prompt.
    """
    assinatura: List[Dict[str, str]] = []
    if not codigo:
        return assinatura

    padrao_input = re.compile(
        r"\b(?:(int|float|str|bool)\s*\(\s*)?input\s*\(",
        flags=re.S,
    )
    padrao_prompt = re.compile(
        r"""input\s*\(\s*(['"])((?:\\.|(?!\1).)*)\1""",
        flags=re.S,
    )
    padrao_atribuicao = re.compile(
        r"([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*(?::[^=\n]+)?=\s*$"
    )

    for match in padrao_input.finditer(codigo):
        prefixo_linha = codigo[codigo.rfind("\n", 0, match.start()) + 1:match.start()]
        variavel = ""
        atrib = padrao_atribuicao.search(prefixo_linha)
        if atrib:
            variavel = re.sub(r"\s+", "", atrib.group(1)).replace(",", ", ")

        prompt = ""
        input_pos = codigo.find("input", match.start(), match.end())
        prompt_match = padrao_prompt.search(codigo, input_pos if input_pos >= 0 else match.start())
        if prompt_match and prompt_match.start() == (input_pos if input_pos >= 0 else match.start()):
            try:
                literal = f"{prompt_match.group(1)}{prompt_match.group(2)}{prompt_match.group(1)}"
                prompt = _normalizar_prompt(ast.literal_eval(literal))
            except Exception:
                prompt = _normalizar_prompt(prompt_match.group(2))

        assinatura.append({
            "indice": str(len(assinatura) + 1),
            "variavel": variavel,
            "prompt": prompt,
            "conversor": match.group(1) or "",
        })

    return assinatura


def extrair_assinatura_inputs(codigo: str) -> List[Dict[str, str]]:
    """
    Extrai a assinatura dos input() em ordem determinística de execução textual.

    Cada item contém:
    - indice: posição 1-based da chamada input()
    - variavel: nome associado por atribuição, quando detectável
    - prompt: prompt literal passado ao input(), quando houver
    - conversor: int/float/str/bool quando o padrão simples for detectado

    Se o código tiver erro de sintaxe, usa um fallback textual conservador
    para ainda recuperar a interface de códigos parcialmente incorretos.
    """
    if not codigo:
        return []

    try:
        arvore = ast.parse(codigo)
    except SyntaxError:
        return _extrair_assinatura_inputs_fallback(codigo)

    visitor = _AssinaturaInputsVisitor()
    visitor.visit(arvore)
    return visitor.assinatura


def extrair_prompts_input(codigo: str) -> List[str]:
    """
    Extrai apenas os prompts literais passados para input("...").

    A função é propositalmente conservadora:
    - só coleta argumentos literais do input()
    - não tenta adivinhar prompts montados dinamicamente
    - não coleta strings de print(), comentário, etc.
    """
    return _unique([item["prompt"] for item in extrair_assinatura_inputs(codigo) if item.get("prompt")])


def remover_prompts_saida(texto: str, prompts_input: Iterable[str]) -> str:
    """
    Remove da saída apenas os prompts capturados por input().

    Importante:
    - não remove outras mensagens do programa
    - não remove o que vem depois do prompt na mesma linha
    - é útil para comparar saída esperada e saída real quando o LLM
      coloca o texto do input() na resposta esperada
    """
    if texto is None:
        return ""
    texto = normalizar_texto(texto)

    prompts = []
    if isinstance(prompts_input, str):
        prompts = [prompts_input]
    else:
        prompts = list(prompts_input or [])

    prompts = _unique([_normalizar_prompt(p) for p in prompts if p])
    if not prompts:
        return texto

    for prompt in sorted(prompts, key=len, reverse=True):
        variants = _unique([
            prompt,
            prompt.rstrip(),
            prompt.strip(),
        ])
        for variante in variants:
            if not variante:
                continue
            # Remove apenas a ocorrência literal do prompt.
            texto = texto.replace(variante, "")

    # Limpeza leve após a remoção.
    texto = re.sub(r"[ \t]+\n", "\n", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return normalizar_texto(texto)
