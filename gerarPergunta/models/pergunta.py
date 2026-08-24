# =============================================================
# models/pergunta.py
# Modelo de dados que representa uma pergunta gerada pela IA
# =============================================================

from dataclasses import dataclass


@dataclass
class Pergunta:
    """
    Representa uma pergunta gerada pelo tutor de IA.

    Contrato de GERAÇÃO (Fase 3.1) — preserva a origem da pergunta:
        tipo             – categoria da pergunta (correcao, justificativa, etc.)
        pergunta         – texto da pergunta gerada
        exercicio_numero – número da questão de origem
        enunciado_origem – enunciado original usado para gerar a pergunta
        codigo_aluno_origem – código produzido pelo aluno que originou a pergunta

        Campos opcionais reservados para etapas posteriores (Fases 3.2+).
        Permanecem None/vazios enquanto não houver mecanismo real de
        preenchimento — o sistema NÃO inventa evidência:
            conceito_avaliado, evidencia_codigo, resposta_esperada, rubrica
    """
    tipo: str
    pergunta: str
    exercicio_numero: int = 0
    enunciado_origem: str = ""
    codigo_aluno_origem: str = ""
    conceito_avaliado: str | None = None
    evidencia_codigo: str | None = None
    resposta_esperada: str | None = None
    rubrica: str | None = None

    def __str__(self):
        return f"[{self.tipo.upper()}] {self.pergunta}"
