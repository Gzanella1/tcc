import json
import os
import shutil
import textwrap
import urllib.request
from json import JSONDecodeError
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

ARQUIVO_HISTORICO = BASE_DIR / "historico.json"
ARQUIVO_CONHECIMENTO = ROOT_DIR / "conteudo" / "conhecimento.txt"

LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "local-model")

MAX_HIST = 10
TEMPERATURA = 0.35

def carregar_conhecimento():
    if ARQUIVO_CONHECIMENTO.exists():
        return ARQUIVO_CONHECIMENTO.read_text(encoding="utf-8")

    print(f"Aviso: base de conhecimento não encontrada em {ARQUIVO_CONHECIMENTO}")
    return ""


def criar_prompt_sistema(conhecimento):
    return {
        "role": "system",
        "content": f"""
Você é um tutor socrático de programação para alunos iniciantes.

Seu objetivo é guiar o aluno, não resolver por ele.
A base de conhecimento abaixo é apenas um gabarito interno. Mesmo que ela tenha respostas completas, você nunca deve copiar, revelar ou adaptar o código final para o aluno.

Base de conhecimento:
{conhecimento}

Regras obrigatórias:
- Comece sempre com uma pergunta.
- Dê somente uma dica pequena por resposta.
- Nunca escreva código completo.
- Nunca entregue a resposta final.
- Nunca use bloco de código, Markdown, listas numeradas ou listas com bullets.
- Se o aluno pedir a resposta pronta, recuse de forma breve e continue guiando.
- Ajude o aluno a encontrar o erro com perguntas socráticas.
- Responda em português do Brasil.
- Mantenha cada bloco com no máximo duas frases.

Formato obrigatório, exatamente nesta ordem:
Pergunta:
<uma pergunta curta para o aluno pensar>

Dica:
<uma dica leve, sem código pronto>

Motivação:
<uma frase curta incentivando o aluno a tentar o próximo passo>
""".strip(),
    }


def carregar_historico():
    if not ARQUIVO_HISTORICO.exists():
        return []

    try:
        dados = json.loads(ARQUIVO_HISTORICO.read_text(encoding="utf-8"))
    except (JSONDecodeError, OSError):
        print("Aviso: histórico inválido. Começando uma nova conversa.")
        return []

    if not isinstance(dados, list):
        return []

    return [
        mensagem
        for mensagem in dados
        if isinstance(mensagem, dict)
        and mensagem.get("role") in {"user", "assistant"}
        and isinstance(mensagem.get("content"), str)
    ]


def salvar_historico(history):
    ARQUIVO_HISTORICO.write_text(
        json.dumps(history, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )


def montar_mensagens(conhecimento, history, user_input):
    mensagens = [criar_prompt_sistema(conhecimento)]
    mensagens.extend(history[-MAX_HIST:])
    mensagens.append({"role": "user", "content": user_input})
    return mensagens


def chamar_tutor(messages):
    payload = {
        "model": LM_STUDIO_MODEL,
        "messages": messages,
        "temperature": TEMPERATURA,
        "max_tokens": 350,
    }
    dados = json.dumps(payload).encode("utf-8")
    url = LM_STUDIO_BASE_URL.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=dados,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LM_STUDIO_API_KEY}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as response:
        raw = response.read().decode("utf-8", errors="replace")

    resposta = json.loads(raw)
    return resposta["choices"][0]["message"]["content"].strip()


def parece_ter_codigo(texto):
    gatilhos = (
        "```",
        "print(",
        "input(",
        "range(",
        "len(",
        "def ",
        "class ",
        "import ",
        "return ",
        "for ",
        "while ",
        "if ",
        "else:",
        "elif ",
    )

    linhas = [linha.strip() for linha in texto.splitlines()]
    return any(linha.startswith(gatilhos) for linha in linhas)


def esta_no_formato(texto):
    normalizado = texto.lower()
    return (
        "pergunta:" in normalizado
        and "dica:" in normalizado
        and ("motivação:" in normalizado or "motivacao:" in normalizado)
    )


def resposta_fallback():
    return "\n\n".join(
        [
            "Pergunta:\nO que você já tentou fazer e em qual parte ficou travado?",
            "Dica:\nQuebre o problema em um passo bem pequeno antes de pensar no código inteiro.",
            "Motivação:\nTente esse primeiro passo e me mostre o resultado.",
        ]
    )


def revisar_resposta(conteudo):
    if esta_no_formato(conteudo) and not parece_ter_codigo(conteudo):
        return conteudo

    mensagens = [
        {
            "role": "system",
            "content": """
Reescreva a resposta para obedecer exatamente ao formato:
Pergunta:
Dica:
Motivação:

Não use código. Não entregue resposta final. Use português do Brasil.
""".strip(),
        },
        {"role": "user", "content": conteudo},
    ]

    try:
        revisada = chamar_tutor(mensagens)
    except Exception:
        return resposta_fallback()

    if esta_no_formato(revisada) and not parece_ter_codigo(revisada):
        return revisada

    return resposta_fallback()


def largura_terminal():
    return min(shutil.get_terminal_size((88, 20)).columns, 100)


def formatar_texto(texto):
    largura = largura_terminal()
    linhas_formatadas = []

    for bloco in texto.splitlines():
        bloco = bloco.strip()

        if not bloco:
            linhas_formatadas.append("")
            continue

        if bloco.lower().rstrip(":") in {"pergunta", "dica", "motivacao", "motivação"}:
            linhas_formatadas.append(bloco)
            continue

        linhas_formatadas.append(
            textwrap.fill(
                bloco,
                width=largura - 4,
                replace_whitespace=True,
                drop_whitespace=True,
            )
        )

    return "\n".join(linhas_formatadas)


def imprimir_tutor(texto):
    largura = largura_terminal()
    print()
    print("-" * largura)
    print("Tutor".center(largura))
    print("-" * largura)
    print(formatar_texto(texto))
    print("-" * largura)


def imprimir_inicio():
    largura = largura_terminal()
    print("-" * largura)
    print("Tutor de programação".center(largura))
    print("-" * largura)
    print("Digite 'sair' para encerrar ou 'limpar' para apagar o histórico.")


def limpar_historico():
    if ARQUIVO_HISTORICO.exists():
        ARQUIVO_HISTORICO.unlink()
    print("Histórico limpo. Vamos recomeçar.")


def main():
    conhecimento = carregar_conhecimento()
    history = carregar_historico()

    imprimir_inicio()

    while True:
        user_input = input("\nAluno > ").strip()

        if not user_input:
            continue

        comando = user_input.lower()

        if comando == "sair":
            break

        if comando == "limpar":
            history = []
            limpar_historico()
            continue

        mensagens = montar_mensagens(conhecimento, history, user_input)

        try:
            content = chamar_tutor(mensagens)
            content = revisar_resposta(content)
        except Exception as e:
            print(f"\nErro ao chamar o modelo: {e}")
            continue

        imprimir_tutor(content)

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": content})
        salvar_historico(history)


if __name__ == "__main__":
    main()
