# =============================================================
# services/question_orchestrator.py
# Responsabilidade única: coordenar o fluxo de geração de
# perguntas — decide quantos exercícios usar, sorteia os tipos
# e chama o AIClient para cada par (exercício, tipo).
# =============================================================

from models.exercicio import Exercicio
from models.pergunta   import Pergunta
from services.ai_client import AIClient
from utils.sorteador    import Sorteador
from config.settings    import TOTAL_PERGUNTAS


class QuestionOrchestrator:
    """
    Orquestra a geração das perguntas.

    Fluxo:
        1. Recebe a lista de exercícios carregados
        2. Decide quais exercícios serão usados (repete se necessário)
        3. Sorteia um tipo de pergunta para cada slot
        4. Chama AIClient para cada par (exercício, tipo)
        5. Retorna lista de (exercicio, pergunta)
    """

    def __init__(self, ai_client: AIClient | None = None):
        self._ai   = ai_client or AIClient()
        self._sort = Sorteador()

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def gerar(
        self,
        exercicios: list[Exercicio],
        total: int = TOTAL_PERGUNTAS,
    ) -> list[tuple[Exercicio, Pergunta]]:
        """
        Gera `total` pares (Exercicio, Pergunta).

        Parâmetros:
            exercicios – lista de exercícios disponíveis
            total      – quantidade de perguntas a gerar

        Retorna:
            lista de tuplas (Exercicio, Pergunta)
        """
        if not exercicios:
            raise ValueError("Lista de exercícios está vazia.")

        slots     = self._montar_slots(exercicios, total)
        tipos     = self._sort.tipos_aleatorios(total)
        resultado = []

        for idx, (exercicio, tipo) in enumerate(zip(slots, tipos), start=1):
            print(f"  ➜  Gerando pergunta {idx}/{total} "
                  f"[{tipo}] — Exercício {exercicio.numero}")

            perguntas = self._ai.gerar_pergunta(exercicio, tipo)

            for p in perguntas:
                resultado.append((exercicio, p))

        return resultado

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _montar_slots(
        self,
        exercicios: list[Exercicio],
        total: int,
    ) -> list[Exercicio]:
        """
        Retorna uma lista de tamanho `total` com exercícios.
        Se houver menos exercícios do que slots, repete em ciclo.
        """
        qtd = len(exercicios)
        if qtd >= total:
            return exercicios[:total]

        slots = []
        for i in range(total):
            slots.append(exercicios[i % qtd])
        return slots
