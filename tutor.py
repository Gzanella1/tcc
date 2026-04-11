import json
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

ARQUIVO_HISTORICO = "historico.json"
ARQUIVO_CONHECIMENTO = "conhecimento.txt"
MAX_HIST = 10  # 🔥 Limita histórico para performance

# 🔹 Carregar base de conhecimento
def carregar_conhecimento():
    if os.path.exists(ARQUIVO_CONHECIMENTO):
        with open(ARQUIVO_CONHECIMENTO, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# 🔹 Criar prompt do sistema
def criar_prompt_sistema(conhecimento):
    return {
        "role": "system",
        "content": f"""
Você é um TUTOR DE PROGRAMAÇÃO INTERATIVO.
Não faça o código, se pedir responda fazendo perguntas 

Você NUNCA resolve o problema para o aluno.
Você NUNCA escreve código completo.
Você NUNCA dá a resposta final.

Seu papel é ENSINAR, não resolver.

Base de conhecimento:
{conhecimento}

REGRAS CRÍTICAS (OBRIGATÓRIO):
- Sempre faça pelo menos 1 pergunta antes de qualquer explicação
- Sempre dê apenas uma dica pequena
- Nunca entregue código completo
- Nunca responda diretamente
- Se o aluno pedir a resposta, recuse e continue guiando
- Ajude o aluno a encontrar o erro com perguntas socráticas.

FORMATO OBRIGATÓRIO:
Pergunta:
<pergunta>

Dica:
<dica leve>

Motivação:
<tente novamente>

Se você não seguir isso, sua resposta está errada.
"""
    }

# 🔹 Carregar histórico
def carregar_historico(conhecimento):
    if os.path.exists(ARQUIVO_HISTORICO):
        with open(ARQUIVO_HISTORICO, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return [criar_prompt_sistema(conhecimento)]

# 🔹 Salvar histórico
def salvar_historico(history):
    with open(ARQUIVO_HISTORICO, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

# 🔹 Inicialização
conhecimento = carregar_conhecimento()
history = carregar_historico(conhecimento)

print("🎓 Tutor baseado em respostas do aluno! (digite 'sair')")

while True:
    user_input = input("\nAluno: ")

    if user_input.lower() == "sair":
        break

    history.append({"role": "user", "content": user_input})

    # 🔥 Limita histórico (mantém system + últimas mensagens)
    system_prompt = history[0]
    mensagens_recentes = history[-MAX_HIST:]
    history_limitado = [system_prompt] + mensagens_recentes

    try:
        response = client.chat.completions.create(
            model="local-model",
            messages=history_limitado,
            temperature=0.7
        )

        content = response.choices[0].message.content

        print("\nTutor:", content)

        history.append({"role": "assistant", "content": content})

        salvar_historico(history)

    except Exception as e:
        print(f"\nErro: {e}")