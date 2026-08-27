#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/generator.py

Geração, validação e deduplicação de casos de teste para questões de código.
Os testes podem vir do próprio enunciado (explícitos) ou ser gerados via LLM.

Contrato canônico:
- codigo_base          : interface/estrutura do programa anterior do aluno
                         (apenas contexto; NUNCA tratado como gabarito).
- codigo_aluno_resposta: o código a ser corrigido. NUNCA participa da
                         geração da própria régua (anti-autocircularidade):
                         ele não gera saída esperada nem define testes que
                         o avaliarão. Porém, para MODIFICACAO/CORRECAO, a
                         interface declarada nele é usada para:
                         - detectar se o código usa input()
                         - determinar a quantidade/orde de inputs
                         - extrair prompts para limpeza de entradas
- entradaTestes/saidaTestes/saida_esperada/testes: fontes explícitas de régua.

Princípio: o ENUNCIADO é a fonte primária do comportamento esperado.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from config import TESTES_ALVO, USAR_LLM
from llm.client import chamar_llm_json
from models.questao import Questao
from utils.text import (
    blocos_na_entrada,
    codigo_tem_input,
    extrair_assinatura_inputs,
    extrair_prompts_input,
    normalizar_label,
    normalizar_texto,
)

# ─── Etapa 4.4: procedência da régua de testes ────────────────────────────────
# Chave INTERNA "_origem" nos dicionários de teste; não faz parte da interface
# pública documentada (entrada/saida/obs). Valores válidos: "enunciado" | "llm".
ORIGEM_ENUNCIADO = "enunciado"
ORIGEM_LLM = "llm"
_CHAVE_ORIGEM = "_origem"


def _com_origem(teste: Dict[str, str], origem: str) -> Dict[str, str]:
    """Copia um dicionário de teste adicionando a procedência interna."""
    novo = dict(teste)
    novo[_CHAVE_ORIGEM] = origem
    return novo


def _codigo_para_interface(q: Questao) -> str:
    """
    Retorna o código que define a interface atual do programa.

    Para MODIFICACAO e CORRECAO: usa codigo_aluno_resposta (a interface
    atual que o código implementa), com fallback para codigo_base.

    Para outros tipos: usa codigo_base.

    A interface determina:
    - se o código usa input()
    - quantos inputs são necessários
    - quais prompts os inputs exibem
    """
    # Para MODIFICACAO/CORRECAO, a interface atual é a do código do aluno.
    if q.tipo in ("modificacao", "correcao"):
        if q.codigo_aluno_resposta and q.codigo_aluno_resposta.strip():
            return q.codigo_aluno_resposta
    return q.codigo_base or ""


def _descrever_assinatura_inputs(assinatura: List[Dict[str, str]]) -> str:
    """Formata a assinatura de input() para orientar a geração estruturada."""
    if not assinatura:
        return "(nenhum input() detectado de forma estática)"

    linhas = []
    for item in assinatura:
        indice = item.get("indice", "")
        variavel = item.get("variavel") or "(sem variável detectada)"
        prompt = item.get("prompt") or "(sem prompt literal)"
        conversor = item.get("conversor") or "(sem conversor detectado)"
        linhas.append(
            f"{indice}. variavel={variavel!r}; prompt={prompt!r}; conversor={conversor!r}"
        )
    return "\n".join(linhas)


def _normalizar_valor_stdin(valor: Any) -> str:
    """Converte um valor de caso em uma única linha de stdin."""
    if valor is None:
        return ""

    texto = normalizar_texto(str(valor))
    if not texto:
        return ""

    linhas = [linha.strip() for linha in texto.splitlines() if linha.strip()]
    if not linhas:
        return ""
    if len(linhas) == 1:
        return linhas[0]
    return " ".join(linhas)


def _indice_entrada(valor: Any) -> Optional[int]:
    """Extrai índices 1-based de chaves como 1, "1", "input_1" ou "entrada 1"."""
    if isinstance(valor, int):
        return valor if valor > 0 else None

    texto = normalizar_label(str(valor or ""))
    match = re.fullmatch(r"(?:(?:input|entrada|valor|indice|posicao|ordem)\s*)?(\d+)", texto)
    if not match:
        return None

    try:
        indice = int(match.group(1))
    except Exception:
        return None

    return indice if indice > 0 else None


def _chaves_da_assinatura(item: Dict[str, str]) -> List[str]:
    chaves = [
        item.get("variavel", ""),
        item.get("prompt", ""),
        f"input {item.get('indice', '')}",
        f"entrada {item.get('indice', '')}",
        str(item.get("indice", "")),
    ]
    return [normalizar_label(chave) for chave in chaves if normalizar_label(chave)]


def _valor_de_registro(registro: Dict[str, Any]) -> Any:
    for chave in ("valor", "value", "dado", "entrada_valor"):
        if chave in registro:
            return registro[chave]
    return None


def _registrar_valor_estruturado(
    valores_por_indice: Dict[int, str],
    valores_por_chave: Dict[str, str],
    *,
    valor: Any,
    indice: Any = None,
    chaves: List[Any] | None = None,
) -> bool:
    valor_norm = _normalizar_valor_stdin(valor)
    if valor_norm == "":
        return False

    registrou = False
    indice_norm = _indice_entrada(indice)
    if indice_norm is not None:
        valores_por_indice[indice_norm] = valor_norm
        registrou = True

    for chave in chaves or []:
        chave_norm = normalizar_label(str(chave or ""))
        if not chave_norm:
            continue
        valores_por_chave[chave_norm] = valor_norm
        indice_da_chave = _indice_entrada(chave)
        if indice_da_chave is not None:
            valores_por_indice[indice_da_chave] = valor_norm
        registrou = True

    return registrou


def _coletar_valores_estruturados(item: Dict[str, Any]) -> tuple[bool, Dict[int, str], Dict[str, str]]:
    """
    Coleta valores de entrada em formatos estruturados.

    Aceita tanto o formato preferencial:
        "valores_entrada": [{"variavel": "opcao", "valor": "1"}, ...]
    quanto formatos defensivos por dicionário:
        "valores_entrada": {"opcao": "1", "a": "5"}
    """
    valores_por_indice: Dict[int, str] = {}
    valores_por_chave: Dict[str, str] = {}
    encontrou_estrutura = False

    fontes = []
    for chave in (
        "valores_entrada",
        "inputs",
        "entradas",
        "valores_por_input",
        "valores_por_variavel",
    ):
        if chave in item:
            fontes.append(item[chave])

    entrada = item.get("entrada")
    if isinstance(entrada, (dict, list)):
        fontes.append(entrada)

    def registrar_registro(registro: Dict[str, Any], chave_externa: Any = None) -> bool:
        valor = _valor_de_registro(registro)
        if valor is None:
            return False

        chaves = [
            registro.get("variavel"),
            registro.get("var"),
            registro.get("nome"),
            registro.get("campo"),
            registro.get("prompt"),
            registro.get("label"),
            registro.get("chave"),
        ]
        indice = (
            registro.get("indice")
            or registro.get("index")
            or registro.get("ordem")
            or registro.get("posicao")
            or registro.get("posição")
            or registro.get("input")
        )

        tem_chave_semantica = any(normalizar_label(str(chave or "")) for chave in chaves)

        if chave_externa is not None:
            if isinstance(chave_externa, int):
                if indice is None and not tem_chave_semantica:
                    indice = chave_externa
            else:
                chaves.insert(0, chave_externa)

        return _registrar_valor_estruturado(
            valores_por_indice,
            valores_por_chave,
            valor=valor,
            indice=indice,
            chaves=chaves,
        )

    for fonte in fontes:
        if isinstance(fonte, dict):
            encontrou_estrutura = True
            if registrar_registro(fonte):
                continue

            for chave, valor in fonte.items():
                if isinstance(valor, dict):
                    registrar_registro(valor, chave_externa=chave)
                else:
                    _registrar_valor_estruturado(
                        valores_por_indice,
                        valores_por_chave,
                        valor=valor,
                        indice=chave,
                        chaves=[chave],
                    )

        elif isinstance(fonte, list):
            encontrou_estrutura = True
            for posicao, valor in enumerate(fonte, start=1):
                if isinstance(valor, dict):
                    registrar_registro(valor, chave_externa=posicao)
                else:
                    _registrar_valor_estruturado(
                        valores_por_indice,
                        valores_por_chave,
                        valor=valor,
                        indice=posicao,
                    )

    return encontrou_estrutura, valores_por_indice, valores_por_chave


def _montar_entrada_por_assinatura(
    item: Dict[str, Any],
    assinatura: List[Dict[str, str]],
) -> Optional[str]:
    """
    Monta stdin com a ordem real dos input() detectada no código.

    O LLM fornece valores do cenário; a sequência final é decidida aqui.
    """
    if not assinatura:
        return None

    encontrou_estrutura, valores_por_indice, valores_por_chave = _coletar_valores_estruturados(item)
    if not encontrou_estrutura:
        return None

    linhas: List[str] = []
    for entrada in assinatura:
        valor = None
        for chave in _chaves_da_assinatura(entrada):
            valor = valores_por_chave.get(chave)
            if valor is not None:
                break

        if valor is None:
            indice = _indice_entrada(entrada.get("indice"))
            valor = valores_por_indice.get(indice) if indice is not None else None

        if valor is None:
            return None

        linhas.append(valor)

    return "\n".join(linhas) + "\n"


def _montar_entrada_gerada(
    item: Dict[str, Any],
    assinatura: List[Dict[str, str]],
    interface: str,
) -> str:
    entrada = _montar_entrada_por_assinatura(item, assinatura)
    if entrada is not None:
        return entrada

    entrada_bruta = item.get("entrada", "")
    if isinstance(entrada_bruta, (dict, list)):
        return ""

    return limpar_entrada_interativa(str(entrada_bruta), interface)

# ─── Deduplicação ────────────────────────────────────────────────────────────

def deduplicar_testes(testes: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove testes com entradas duplicadas, mantendo a primeira ocorrência."""
    vistos: set = set()
    saida: List[Dict[str, str]] = []

    for t in testes:
        entrada = normalizar_texto(str(t.get("entrada", "")))
        if entrada in vistos:
            continue
        vistos.add(entrada)
        novo = {
            "entrada": entrada,
            "saida":   normalizar_texto(str(t.get("saida", ""))),
            "obs":     normalizar_texto(str(t.get("obs", ""))),
        }
        # Etapa 4.4: a procedência acompanha o teste que sobrevive à
        # deduplicação (a regra de qual teste sobrevive permanece a mesma).
        if t.get(_CHAVE_ORIGEM):
            novo[_CHAVE_ORIGEM] = str(t[_CHAVE_ORIGEM])
        saida.append(novo)

    return saida


# ─── Validação ───────────────────────────────────────────────────────────────

def validar_testes(testes: List[Dict], requer_input: bool = False) -> List[Dict[str, str]]:
    """
    Filtra testes inválidos.

    Regras:
    - entrada e saida não podem ser None
    - saida não pode ser vazia
    - se o código usa input(), entrada vazia é descartada
    """
    validos = []

    for t in testes:
        if not isinstance(t, dict):
            continue

        entrada = t.get("entrada")
        saida = t.get("saida")

        if entrada is None or saida is None:
            continue

        entrada = normalizar_texto(str(entrada))
        saida = normalizar_texto(str(saida))
        obs = normalizar_texto(str(t.get("obs", "")))

        if saida == "":
            continue

        if requer_input and entrada == "":
            continue

        novo = {
            "entrada": entrada,
            "saida": saida,
            "obs": obs,
        }
        # Etapa 4.4: preserva a procedência interna na reconstrução.
        if t.get(_CHAVE_ORIGEM):
            novo[_CHAVE_ORIGEM] = str(t[_CHAVE_ORIGEM])
        validos.append(novo)

    return validos


# ─── Geração via LLM ─────────────────────────────────────────────────────────

def _gerar_testes_llm_once(q: Questao, quantidade: int) -> List[Dict[str, str]]:
    """
    Faz uma única chamada ao LLM pedindo {quantidade} testes para a questão.
    Retorna a lista de testes validados gerada.
    """
    # Interface: codigo_aluno_resposta (MODIFICACAO/CORRECAO) ou codigo_base.
    interface = _codigo_para_interface(q)
    usa_input = codigo_tem_input(interface)
    assinatura_inputs = extrair_assinatura_inputs(interface)
    assinatura_descrita = _descrever_assinatura_inputs(assinatura_inputs)

    if usa_input:
        regra_input = "- O código usa input(), então gere valores realistas para cada input()."
        regra_entrada = f"""
- NÃO defina a ordem final do stdin na chave "entrada"; ela será montada pelo Python.
- Preencha "valores_entrada" com os valores do cenário, identificados por indice/variavel/prompt.
- A ordem dos objetos dentro de "valores_entrada" não importa.
- O Python montará "entrada" seguindo exatamente esta assinatura real dos input():
{assinatura_descrita}
- NÃO inclua textos dos prompts, menus ou mensagens do programa como valores.
- Exemplo de valores_entrada:
  [
    {{"indice": 1, "variavel": "opcao", "valor": "1"}},
    {{"indice": 2, "variavel": "a", "valor": "5"}},
    {{"indice": 3, "variavel": "b", "valor": "3"}}
  ]
"""
    else:
        regra_input = "- O código NÃO usa input(), então NÃO gere entradas."
        regra_entrada = '- entrada deve ser "" (string vazia)'

    prompt = f"""
Você é um gerador de testes para código Python.

Tipo da questão:
{q.tipo}

IMPORTANTE:
{regra_input}
- A saída deve refletir o comportamento correto do programa conforme o ENUNCIADO
- O ENUNCIADO é a fonte PRIMÁRIA para determinar o comportamento esperado
- Se o enunciado pedir uma funcionalidade nova (modificação), teste o programa FINAL como descrito no enunciado
- Se o enunciado pedir saída adicional, ela deve aparecer na saída esperada
- NÃO use o código-base para definir a saída esperada — ele é apenas contexto para entender o problema original
- Não invente comportamento fora do enunciado
- Respeite rigorosamente o funcionamento real esperado

Enunciado (fonte dos requisitos):
{q.enunciado}

Código-base (código ANTERIOR do aluno — apenas contexto para entender o problema original; NÃO é gabarito e NÃO define a saída esperada):
{q.codigo_base or "(não fornecido)"}

_INTERFACE_ATUAL_ (interface implementada — define quantos inputs e em que ordem; NÃO define a saída esperada):
{interface or "(não disponível)"}

Gere exatamente {quantidade} testes.

Formato obrigatório (JSON puro):
{{
  "testes": [
    {{
      "entrada": "",
      "valores_entrada": [
        {{"indice": 1, "variavel": "nome_da_variavel", "prompt": "prompt literal se houver", "valor": "valor digitado"}}
      ],
      "saida": "saida esperada\\n",
      "obs": "descrição do caso"
    }}
  ]
}}

REGRAS:
{regra_entrada}
- saida deve terminar com \\n
- não escreva nada fora do JSON
"""

    obj = chamar_llm_json(
        [
            {"role": "system", "content": "Você gera testes válidos e retorna apenas JSON puro."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1200,
    )

    testes: List[Dict[str, str]] = []

    if isinstance(obj, dict):
        bruto = obj.get("testes", [])
        if isinstance(bruto, list):
            for item in bruto:
                if isinstance(item, dict):
                    entrada = _montar_entrada_gerada(item, assinatura_inputs, interface)
                    saida = item.get("saida", "")

                    if not usa_input:
                        entrada = ""

                    if (
                        usa_input
                        and assinatura_inputs
                        and blocos_na_entrada(entrada) != len(assinatura_inputs)
                    ):
                        continue

                    testes.append({
                        "entrada": str(entrada),
                        "saida": str(saida),
                        "obs": str(item.get("obs", "")),
                        "_origem": ORIGEM_LLM,
                    })

    return validar_testes(testes, requer_input=usa_input)


def gerar_testes_com_llm(q: Questao, quantidade: int = TESTES_ALVO) -> List[Dict[str, str]]:
    """
    Gera testes via LLM com até 3 tentativas.
    Retorna assim que obtiver ao menos 1 teste válido.
    Caso contrário, retorna o melhor resultado obtido.
    """
    if not USAR_LLM:
        return []

    melhor: List[Dict[str, str]] = []

    for _ in range(3):
        testes = _gerar_testes_llm_once(q, quantidade)

        if len(testes) > len(melhor):
            melhor = testes

        if len(testes) >= 1:
            return testes

    return melhor


# ─── Interface pública ────────────────────────────────────────────────────────

def obter_testes_explicitos(q: Questao) -> List[Dict[str, str]]:
    """Retorna os testes declarados explicitamente no enunciado ou nos campos da questão."""
    testes: List[Dict[str, str]] = []

    entrada = normalizar_texto(str(q.entradaTestes or ""))
    saida = normalizar_texto(str(q.saidaTestes or q.saida_esperada or ""))

    # Interface: codigo_aluno_resposta (MODIFICACAO/CORRECAO) ou codigo_base.
    interface = _codigo_para_interface(q)

    if saida:
        if codigo_tem_input(interface):
            if entrada != "":
                testes.append({
                    "entrada": entrada,
                    "saida": saida,
                    "obs": "Caso explícito do enunciado",
                    "_origem": ORIGEM_ENUNCIADO,
                })
        else:
            testes.append({
                "entrada": entrada,
                "saida": saida,
                "obs": "Caso explícito do enunciado",
                "_origem": ORIGEM_ENUNCIADO,
            })

    if q.testes:
        # Testes vindos dos próprios dados da questão também são do enunciado.
        # Cópia para não mutar os dicionários originais da Questao.
        testes.extend(
            _com_origem(dict(t), ORIGEM_ENUNCIADO) for t in q.testes
        )

    return deduplicar_testes(testes)


def obter_testes(q: Questao) -> List[Dict[str, str]]:
    """
    Retorna a lista final de testes para uma questão, combinando:
    - testes explícitos do enunciado
    - testes gerados via LLM

    Também corrige entradas interativas geradas incorretamente pelo LLM.
    """
    testes = obter_testes_explicitos(q)

    # Interface: codigo_aluno_resposta (MODIFICACAO/CORRECAO) ou codigo_base.
    # A interface determina se o código usa input() e quais prompts exibe.
    interface = _codigo_para_interface(q)
    usa_input = codigo_tem_input(interface)

    if usa_input:
        gerados = gerar_testes_com_llm(q, quantidade=TESTES_ALVO)
    else:
        gerados = []

    if gerados:
        testes.extend(gerados)

    # Corrige entradas do tipo:
    # "Digite o primeiro número: 5"
    # para:
    # "5"
    if usa_input:
        testes_corrigidos = []

        for teste in testes:
            teste_corrigido = dict(teste)
            teste_corrigido["entrada"] = limpar_entrada_interativa(
                teste_corrigido.get("entrada", ""),
                interface,
            )
            testes_corrigidos.append(teste_corrigido)

        testes = testes_corrigidos

    testes = validar_testes(testes, requer_input=usa_input)
    testes = deduplicar_testes(testes)

    if not testes:
        return []

    return testes[:TESTES_ALVO]



def limpar_entrada_interativa(entrada: str, codigo: str) -> str:
    """
    Limpa entradas geradas pelo LLM quando ele inclui textos de prompt.

    Exemplo errado gerado pelo LLM:
        Digite o primeiro número: 5
        Digite o segundo número: 3
        Escolha a operação:
        1 - Soma
        2 - Subtração
        3 - Multiplicação
        4 - Divisão
        Digite a opção desejada: 1

    Entrada correta para stdin:
        5
        3
        1
    """
    entrada = normalizar_texto(entrada)

    if not entrada:
        return ""

    prompts = extrair_prompts_input(codigo)

    linhas = [linha.strip() for linha in entrada.splitlines() if linha.strip()]
    valores = []

    for linha in linhas:
        # Ignora linhas de menu, como:
        # 1 - Soma
        # 2 - Subtração
        if re.match(r"^\d+\s*[-.)]\s*\D+", linha):
            continue

        # Se a linha começa com algum prompt do input(),
        # remove o prompt e mantém apenas o valor digitado.
        removeu_prompt = False

        for prompt in prompts:
            prompt = normalizar_texto(prompt).strip()

            if prompt and linha.startswith(prompt):
                valor = linha[len(prompt):].strip()

                if valor:
                    valores.append(valor)

                removeu_prompt = True
                break

        if removeu_prompt:
            continue

        # Se a linha tem formato "algum texto: valor",
        # pega apenas o valor depois dos dois-pontos.
        if ":" in linha:
            _, direita = linha.rsplit(":", 1)
            valor = direita.strip()

            if valor:
                valores.append(valor)

            continue

        # Caso a linha já seja um valor puro, mantém.
        valores.append(linha)

    if not valores:
        return ""

    return "\n".join(valores) + "\n"
