# =============================================================
# utils/sorteador.py
# Responsabilidade única: sortear tipos de pergunta de forma
# embaralhada, sem repetir na mesma rodada.
# =============================================================

import random
from config.settings import TIPOS_PERGUNTA


class Sorteador:
    """
    Sorteia tipos de pergunta garantindo que cada tipo apareça
    ao menos uma vez antes de repetir (embaralhamento sem reposição
    dentro de cada bloco de 5).
    """

    def __init__(self, tipos: list[str] | None = None):
        self._tipos = tipos or TIPOS_PERGUNTA.copy()

    def tipos_aleatorios(self, quantidade: int) -> list[str]:
        """
        Retorna uma lista de `quantidade` tipos embaralhados.

        A lista nunca repete um tipo antes de esgotar todos os
        disponíveis (dentro de cada bloco).
        """
        resultado: list[str] = []

        while len(resultado) < quantidade:
            bloco = self._tipos.copy()
            random.shuffle(bloco)
            resultado.extend(bloco)

        return resultado[:quantidade]
