# =============================================================
# services/question_builder.py
# Responsabilidade única: montar o prompt correto para cada
# tipo de pergunta.  Cada tipo tem seu próprio método, tornando
# fácil adicionar novos tipos no futuro.
# =============================================================

from models.exercicio import Exercicio


class QuestionBuilder:
    """
    Constrói o prompt enviado à IA de acordo com o tipo de pergunta.

    Tipos suportados:
        correcao      – detectar erros no código
        justificativa – explicar por que o código funciona
        descritiva    – descrever o que o código faz
        modificacao   – propor uma alteração no código
        previsao      – prever a saída para uma entrada específica
    """

    _RODAPE = (
        "\nRegras obrigatórias:\n"
        "- Não forneça a resposta\n"
        "- Não copie o código completo\n"
        "- Seja claro e objetivo\n\n"
        "Retorne APENAS um JSON válido, sem texto extra:\n"
        '[{{"tipo": "{tipo}", "pergunta": "<texto da pergunta>"}}]'
    )

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def construir(self, exercicio: Exercicio, tipo: str) -> str:
        """Retorna o prompt completo para o par (exercício, tipo)."""
        metodos = {
            "correcao":      self._correcao,
            "justificativa": self._justificativa,
            "descritiva":    self._descritiva,
            "modificacao":   self._modificacao,
            "previsao":      self._previsao,
        }

        metodo = metodos.get(tipo)
        if metodo is None:
            raise ValueError(
                f"Tipo de pergunta desconhecido: '{tipo}'. "
                f"Disponíveis: {list(metodos)}"
            )
        return metodo(exercicio)

    # ------------------------------------------------------------------
    # Cabeçalho e rodapé compartilhados
    # ------------------------------------------------------------------

    def _cabecalho(self, exercicio: Exercicio) -> str:
        """
        Cabeçalho do prompt com o papel semântico explícito de cada dado
        (Fase 3.1): enunciado original + código produzido pelo aluno.
        """
        return (
            f"Exercício {exercicio.numero}.\n\n"
            f"Enunciado original:\n{exercicio.enunciado_original}\n\n"
            f"Código produzido pelo aluno:\n{exercicio.codigo_aluno_anterior}\n"
        )

    def _rodape(self, tipo: str) -> str:
        return self._RODAPE.format(tipo=tipo)

    # ------------------------------------------------------------------
    # Um método por tipo de pergunta
    # ------------------------------------------------------------------

    def _correcao(self, exercicio: Exercicio) -> str:
        return (
            "Você é um tutor de programação.\n\n"
            "Gere 1 pergunta do tipo CORREÇÃO que leve o aluno a identificar "
            "um possível erro lógico, sintático ou de borda no código.\n\n"
            + self._cabecalho(exercicio)
            + self._rodape("correcao")
        )

    def _justificativa(self, exercicio: Exercicio) -> str:
        return (
            "Você é um tutor de programação.\n\n"
            "Gere 1 pergunta do tipo JUSTIFICATIVA que peça ao aluno para "
            "explicar por que escolheu determinada abordagem no código.\n\n"
            + self._cabecalho(exercicio)
            + self._rodape("justificativa")
        )

    def _descritiva(self, exercicio: Exercicio) -> str:
        return (
            "Você é um tutor de programação.\n\n"
            "Gere 1 pergunta do tipo DESCRITIVA que peça ao aluno para "
            "descrever, com suas próprias palavras, o que o código faz passo a passo.\n\n"
            + self._cabecalho(exercicio)
            + self._rodape("descritiva")
        )

    def _modificacao(self, exercicio: Exercicio) -> str:
        return (
            "Você é um tutor de programação.\n\n"
            "Gere 1 pergunta do tipo MODIFICAÇÃO que proponha uma alteração "
            "concreta no código (ex.: nova funcionalidade, refatoração, otimização) "
            "e peça ao aluno para pensar como faria essa mudança.\n\n"
            + self._cabecalho(exercicio)
            + self._rodape("modificacao")
        )

    def _previsao(self, exercicio: Exercicio) -> str:
        return (
            "Você é um tutor de programação.\n\n"
            "Gere 1 pergunta do tipo PREVISÃO que apresente uma entrada "
            "específica e pergunte qual será a saída esperada do programa, "
            "incluindo casos de borda se possível.\n\n"
            + self._cabecalho(exercicio)
            + self._rodape("previsao")
        )
