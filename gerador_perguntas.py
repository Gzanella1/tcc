import json
import re
from functools import lru_cache
from openai import OpenAI
import random


# =========================
# CONFIG LLM STUDIO LOCAL
# =========================
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

MODEL = "qwen/qwen3-vl-4b"
TIPOS_PERGUNTA = ["correcao", "justificativa", "descritiva", "modificacao", "previsao"]
TOTAL_PERGUNTAS = 5



# =========================
# LEITURA E PARSING DO ARQUIVO
# =========================
def carregar_conhecimento(arquivo="conhecimento.txt"):
    """
    Lê um arquivo no formato:

    1. Média aritmética de dois números
    2. Número par ou ímpar
    3. Maior de três números
    4. Contagem de 1 até N

    Resposta 1 -
    ...
    Resposta 2 -
    ...
    Resposta 3 -
    ...
    Resposta 4 -
    ...

    Retorna uma lista de dicionários:
    [
        {
            "numero": 1,
            "titulo": "...",
            "codigo": "..."
        },
        ...
    ]
    """
    try:
        with open(arquivo, "r", encoding="utf-8") as f:
            linhas = f.readlines()

        questoes = {}   # {1: "Média aritmética...", 2: "..."}
        respostas = {}   # {1: "codigo...", 2: "codigo..."}

        numero_resposta_atual = None
        codigo_atual = []
        coletando_codigo = False

        def finalizar_resposta():
            nonlocal numero_resposta_atual, codigo_atual, coletando_codigo

            if numero_resposta_atual is not None:
                codigo = "\n".join(codigo_atual).strip()

                # Se o aluno não escreveu nada, mantém um texto padrão
                if not codigo:
                    codigo = "[O ALUNO NÃO ESCREVEU CÓDIGO]"

                respostas[numero_resposta_atual] = codigo

            numero_resposta_atual = None
            codigo_atual = []
            coletando_codigo = False

        for raw in linhas:
            linha = raw.rstrip("\n").strip()

            # Linha de questão: "1. Alguma coisa"
            match_questao = re.match(r"^\s*(\d+)\.\s*(.+?)\s*$", linha)

            # Linha de resposta: "Resposta 1 -"
            match_resposta = re.match(r"^\s*Resposta\s*(\d+)\s*-\s*$", linha, re.IGNORECASE)

            if match_questao:
                numero = int(match_questao.group(1))
                titulo = match_questao.group(2).strip()
                questoes[numero] = titulo
                continue

            if match_resposta:
                # Se já estava coletando uma resposta anterior, salva antes de começar outra
                if coletando_codigo:
                    finalizar_resposta()

                numero_resposta_atual = int(match_resposta.group(1))
                coletando_codigo = True
                codigo_atual = []
                continue

            # Linhas do código/resposta do aluno
            if coletando_codigo:
                codigo_atual.append(raw.rstrip("\n"))

        # Fecha a última resposta do arquivo
        if coletando_codigo:
            finalizar_resposta()

        if not respostas:
            raise ValueError("Nenhuma resposta válida encontrada no arquivo.")

        # Monta a lista final associando cada resposta ao título correto
        exercicios = []
        for numero in sorted(respostas.keys()):
            exercicios.append({
                "numero": numero,
                "titulo": questoes.get(numero, f"Questão {numero}"),
                "codigo": respostas[numero]
            })

        return exercicios

    except Exception as e:
        print("Erro ao ler arquivo:", e)
        return []



# =========================
# SORTEIA AS QUESTOES SEM REPETIR
# =========================


def gerar_tipos_aleatorios(qtd):
    tipos = TIPOS_PERGUNTA.copy()
    random.shuffle(tipos)

    # Se precisar de mais do que 5, repete embaralhando
    while len(tipos) < qtd:
        extra = TIPOS_PERGUNTA.copy()
        random.shuffle(extra)
        tipos.extend(extra)

    return tipos[:qtd]



# =========================
# PROMPT OTIMIZADO
# =========================
def montar_prompt(numero_exercicio, titulo_exercicio, codigo, tipo_pergunta):
    return f"""
    Você é um tutor de programação.

    Gere exatamente 1 pergunta do tipo "{tipo_pergunta}" sobre este exercício.

    Regras obrigatórias:
    - Não dê resposta
    - Não copie o código
    - Seja claro e objetivo

    Retorne APENAS um JSON válido no formato:
    [
    {{"tipo": "{tipo_pergunta}", "pergunta": "..."}}
    ]

    Exercício {numero_exercicio}:
    {titulo_exercicio}

    Código:
    {codigo}
    """

# =========================
# EXTRAÇÃO SEGURA DO JSON
# =========================
def extrair_json(texto):
    inicio = texto.find("[")
    fim = texto.rfind("]") + 1

    if inicio != -1 and fim != -1:
        try:
            return json.loads(texto[inicio:fim])
        except Exception:
            pass

    return []


# =========================
# CHAMADA AO MODELO
# =========================
@lru_cache(maxsize=100)
def gerar_perguntas_ia(numero_exercicio, titulo_exercicio, codigo, tipo_pergunta):
    prompt = montar_prompt(numero_exercicio, titulo_exercicio, codigo, tipo_pergunta)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Você é um tutor de programação."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )

        texto = response.choices[0].message.content.strip()
        perguntas = extrair_json(texto)

        if not perguntas:
            print("⚠️ Não foi possível interpretar a resposta da IA:")
            print(texto)

        return perguntas

    except Exception as e:
        print("Erro na chamada da IA:", e)
        return []



# =========================
# Salva perguntas no arquivo txt
# =========================


def salvar_perguntas(conteudo, arquivo="conteudo/perguntasGeradas.txt"):
    try:
        with open(arquivo, "w", encoding="utf-8") as f:
            f.write(conteudo)
        print(f"✅ Perguntas salvas em '{arquivo}'")
    except Exception as e:
        print("Erro ao salvar arquivo:", e)


# =========================
# EXECUÇÃO PRINCIPAL
# =========================

def main():
    exercicios = carregar_conhecimento()

    if not exercicios:
        print("Erro ao carregar dados.")
        return

    qtd_exercicios = len(exercicios)

    if qtd_exercicios >= TOTAL_PERGUNTAS:
        lista_exercicios = exercicios[:TOTAL_PERGUNTAS]
    else:
        lista_exercicios = []
        i = 0
        while len(lista_exercicios) < TOTAL_PERGUNTAS:
            lista_exercicios.append(exercicios[i % qtd_exercicios])
            i += 1

    tipos_sorteados = gerar_tipos_aleatorios(len(lista_exercicios))

    # 🔥 Aqui acumulamos tudo
    saida_final = ""

    for idx, (exercicio, tipo) in enumerate(zip(lista_exercicios, tipos_sorteados), 1):
        print("Gerando exercicio",idx)
        numero = exercicio["numero"]
        titulo = exercicio["titulo"]
        codigo = exercicio["codigo"]

        bloco = f"\n========== Exercicio gerado com base na sua resposta da questão {numero} ==========\n\n"
        bloco += f"{titulo}\n\n"

        perguntas = gerar_perguntas_ia(numero, titulo, codigo, tipo)

        for i, p in enumerate(perguntas, 1):
            tipo_txt = p.get("tipo", "desconhecido").upper()
            pergunta = p.get("pergunta", "")
            bloco += f"{idx} - [{tipo_txt}] {pergunta}\n \n \n \n \n \n"


        # imprime no terminal
        #print(bloco)

        # acumula
        saida_final += bloco

    # 💾 salva no arquivo
    salvar_perguntas(saida_final)

# ========================= 
# RODAR
# =========================
if __name__ == "__main__":
    main()