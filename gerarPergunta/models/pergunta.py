# =============================================================
# models/pergunta.py
# Modelo de dados que representa uma pergunta gerada pela IA
# =============================================================

from dataclasses import dataclass


@dataclass
class Pergunta:
    """
    Representa uma pergunta gerada pelo tutor de IA.

    Atributos:
        tipo     – categoria da pergunta (correcao, justificativa, etc.)
        pergunta – texto da pergunta gerada
        exercicio_numero – número da questão de origem
    """
    tipo: str
    pergunta: str
    exercicio_numero: int = 0

    def __str__(self):
        return f"[{self.tipo.upper()}] {self.pergunta}"
