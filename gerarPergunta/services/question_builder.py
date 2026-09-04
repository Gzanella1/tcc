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
            "Código anterior do aluno (contexto/apoio; não é gabarito):\n"
            f"{exercicio.codigo_base}\n"
        )

    def _rodape(self, tipo: str) -> str:
        return self._RODAPE.format(tipo=tipo)

    # ------------------------------------------------------------------
    # Um método por tipo de pergunta
    # ------------------------------------------------------------------


    def _correcao(self, exercicio: Exercicio) -> str:
        return (
            "Você é um tutor de programação especializado em análise rigorosa de código Python.\n\n"

            "Sua tarefa é analisar o código produzido pelo aluno e verificar se existe "
            "um erro REAL que possa originar uma pergunta do tipo CORREÇÃO.\n\n"

            "REGRA PRINCIPAL:\n"
            "NUNCA invente um erro apenas para gerar uma pergunta.\n"
            "Se o código estiver correto, não crie uma falsa afirmação de erro.\n\n"

            "COMO ANALISAR O CÓDIGO:\n"
            "1. Verifique a sintaxe real do código.\n"
            "2. Verifique a lógica e o comportamento do algoritmo.\n"
            "3. Verifique condições de borda.\n"
            "4. Verifique se os identificadores usados são consistentes dentro do próprio código.\n"
            "5. NÃO considere um identificador errado apenas porque poderia ter outro nome.\n"
            "6. Por exemplo, se uma variável chamada 'carater' é criada e usada consistentemente "
            "como 'carater', isso NÃO é erro. O fato de existir a palavra 'caracter' como outra "
            "grafia possível não torna 'carater' incorreto.\n"
            "7. NÃO classifique diferenças de nomenclatura, estilo ou preferência pessoal como erro.\n"
            "8. Considere formas matemáticas e lógicas equivalentes como corretas.\n"
            "9. Só considere algo um erro se puder demonstrar objetivamente, a partir do código, "
            "que ele produz comportamento incorreto, erro de execução, erro de sintaxe ou falha "
            "em algum caso válido.\n\n"

            "VALIDAÇÃO OBRIGATÓRIA:\n"
            "Antes de formular a pergunta, tente refutar a hipótese de erro.\n"
            "Pergunte a si mesmo: 'Esse trecho realmente está incorreto ou apenas poderia "
            "ser escrito de outra maneira?'\n"
            "Se estiver correto, NÃO invente um problema.\n\n"

            "DADOS PARA ANÁLISE:\n"
            f"Enunciado original:\n{exercicio.enunciado_original}\n\n"
            f"Código produzido pelo aluno:\n{exercicio.codigo_base}\n\n"

            "CRITÉRIO PARA GERAR A PERGUNTA:\n"
            "A pergunta somente pode ser gerada se existir um erro real, verificável e "
            "explicável diretamente pelo código.\n\n"

            "Retorne APENAS JSON válido neste formato:\n"
            '[{"tipo": "correcao", "pergunta": "<pergunta baseada em erro real>"}]\n\n'

    
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
