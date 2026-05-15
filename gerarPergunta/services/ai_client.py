# =============================================================
# services/ai_client.py
# Responsabilidade única: se comunicar com a API do LM Studio
# e retornar a lista de Pergunta geradas.
# =============================================================

import json
from functools import lru_cache
from openai import OpenAI

from config.settings import LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY, MODEL, TEMPERATURE
from models.pergunta import Pergunta
from models.exercicio import Exercicio
from services.question_builder import QuestionBuilder


class AIClient:
    """
    Faz a chamada ao modelo de linguagem e converte a resposta
    em objetos Pergunta.

    Utiliza cache LRU para evitar chamadas repetidas com os
    mesmos parâmetros (útil quando há menos exercícios do que
    o total de perguntas pedidas).
    """

    def __init__(self):
        self._client = OpenAI(
            base_url=LM_STUDIO_BASE_URL,
            api_key=LM_STUDIO_API_KEY,
        )
        self._builder = QuestionBuilder()

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def gerar_pergunta(self, exercicio: Exercicio, tipo: str) -> list[Pergunta]:
        """
        Gera uma pergunta do tipo solicitado sobre o exercício.

        Retorna uma lista de Pergunta (normalmente com 1 item).
        """
        prompt = self._builder.construir(exercicio, tipo)
        return self._chamar_api(exercicio.numero, exercicio.titulo, exercicio.codigo, tipo, prompt)

    # ------------------------------------------------------------------
    # Chamada à API com cache
    # ------------------------------------------------------------------

    # O lru_cache não funciona diretamente em métodos de instância com
    # argumentos mutáveis; usamos um wrapper estático + chave hashável.
    def _chamar_api(
        self,
        numero: int,
        titulo: str,
        codigo: str,
        tipo: str,
        prompt: str,
    ) -> list[Pergunta]:
        # Usa uma chave imutável para o cache
        chave = (numero, titulo, codigo, tipo)
        return self._chamar_api_cached(chave, prompt)

    @lru_cache(maxsize=200)
    def _chamar_api_cached(self, chave: tuple, prompt: str) -> list[Pergunta]:
        numero = chave[0]
        tipo   = chave[3]

        try:
            response = self._client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Você é um tutor de programação."},
                    {"role": "user",   "content": prompt},
                ],
                temperature=TEMPERATURE,
            )

            texto = response.choices[0].message.content.strip()
            return self._parsear_resposta(texto, numero)

        except Exception as exc:
            print(f"  ⚠️  Erro na chamada da IA: {exc}")
            return []

    # ------------------------------------------------------------------
    # Parsing da resposta JSON
    # ------------------------------------------------------------------

    def _parsear_resposta(self, texto: str, exercicio_numero: int) -> list[Pergunta]:
        """Extrai o JSON da resposta e converte em objetos Pergunta."""
        dados = self._extrair_json(texto)

        if not dados:
            print(f"  ⚠️  Resposta da IA não contém JSON válido:\n{texto[:200]}")
            return []

        perguntas = []
        for item in dados:
            tipo     = item.get("tipo",     "desconhecido")
            pergunta = item.get("pergunta", "").strip()
            if pergunta:
                perguntas.append(Pergunta(tipo=tipo, pergunta=pergunta, exercicio_numero=exercicio_numero))

        return perguntas

    @staticmethod
    def _extrair_json(texto: str) -> list[dict]:
        """Encontra e parseia o primeiro array JSON na string."""
        inicio = texto.find("[")
        fim    = texto.rfind("]") + 1

        if inicio == -1 or fim == 0:
            return []

        try:
            return json.loads(texto[inicio:fim])
        except json.JSONDecodeError:
            return []
