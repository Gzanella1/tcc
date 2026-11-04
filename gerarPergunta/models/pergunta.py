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
        Campos opcionais reservados para etapas posteriores (Fases 3.2+).
        Permanecem None/vazios enquanto não houver mecanismo real de
        preenchimento — o sistema NÃO inventa evidência:
            conceito_avaliado, evidencia_codigo, resposta_esperada, rubrica
    """
    tipo: str
    pergunta: str
    exercicio_numero: int = 0
    enunciado_origem: str = ""
    conceito_avaliado: str | None = None
    evidencia_codigo: str | None = None
    resposta_esperada: str | None = None
    rubrica: str | None = None
    # Campos intermediários do pipeline de GERAÇÃO de PREVISAO. NÃO fazem
    # parte do contrato da correção (Questao não os expõe): dados_entrada é
    # o dado semântico retornado pelo LLM e entrada_testes é a string de
    # stdin obtida por transformação DETERMINÍSTICA de dados_entrada
    # (services.dados_entrada), validada por execução de referência.
    dados_entrada: dict | list | None = None
    entrada_testes: str = ""

    def __str__(self):
        return f"[{self.tipo.upper()}] {self.pergunta}"
