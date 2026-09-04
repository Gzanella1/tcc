# =============================================================
# models/exercicio.py
# Contrato de GERAÇÃO:
# Um Exercicio carrega o enunciado original da questão e o código
# produzido anteriormente pelo aluno. Esse código cruza a fronteira para
# a correção como codigo_base.
#
# Aliases de compatibilidade:
#   .titulo  ↔  .enunciado_original      (sempre em sincronia)
# =============================================================

from dataclasses import dataclass


_PLACEHOLDER_CODIGO = "[O ALUNO NÃO ESCREVEU CÓDIGO]"

# pares de alias: campo → campo espelho
_ALIASES = {
    "titulo":              "enunciado_original",
    "enunciado_original":  "titulo",
}


@dataclass
class Exercicio:
    """
    Representa uma questão original do conhecimento.txt.

    Atributos:
        numero               – número da questão original
        enunciado_original   – texto original da questão (alias: .titulo)
        codigo_base          – código anterior produzido pelo aluno; contexto
                               que será exportado para a correção.

    O alias titulo/enunciado_original permanece sincronizado em qualquer
    atribuição.
    """

    numero: int
    enunciado_original: str = ""
    codigo_base: str = _PLACEHOLDER_CODIGO
    titulo: str = ""

    def __post_init__(self):
        # Normalização inicial (antes do alias dinâmico ligar)
        if not self.codigo_base:
            object.__setattr__(self, "codigo_base", _PLACEHOLDER_CODIGO)
        if not self.enunciado_original:
            object.__setattr__(self, "enunciado_original", self.titulo)
        if not self.titulo:
            object.__setattr__(self, "titulo", self.enunciado_original)
        object.__setattr__(self, "_inicializado", True)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

        # Alias dinâmico só após a construção completa
        if not getattr(self, "_inicializado", False):
            return

        espelho = _ALIASES.get(name)
        if espelho is not None:
            object.__setattr__(self, espelho, value or "")
