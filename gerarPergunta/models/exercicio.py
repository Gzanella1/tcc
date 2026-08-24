# =============================================================
# models/exercicio.py
# Contrato de GERAÇÃO (Fase 3.1): aqui NÃO existe codigo_base.
# Um Exercicio carrega o enunciado original da questão e o código
# produzido anteriormente pelo aluno — que é a ORIGEM de qualquer
# pergunta gerada. Nunca confundir com os contratos de correção.
#
# Aliases de compatibilidade:
#   .codigo  ↔  .codigo_aluno_anterior   (sempre em sincronia)
#   .titulo  ↔  .enunciado_original      (sempre em sincronia)
# =============================================================

from dataclasses import dataclass


_PLACEHOLDER_CODIGO = "[O ALUNO NÃO ESCREVEU CÓDIGO]"

# pares de alias: campo → campo espelho
_ALIASES = {
    "codigo":              "codigo_aluno_anterior",
    "codigo_aluno_anterior": "codigo",
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
        codigo_aluno_anterior – código que o aluno escreveu na resposta
                                anterior (alias: .codigo)

    Os aliases permanecem sincronizados em qualquer atribuição,
    garantindo compatibilidade com o código legado.
    """

    numero: int
    titulo: str = ""
    codigo: str = _PLACEHOLDER_CODIGO
    enunciado_original: str = ""
    codigo_aluno_anterior: str = ""

    def __post_init__(self):
        # Normalização inicial (antes do alias dinâmico ligar)
        if not self.codigo:
            object.__setattr__(self, "codigo", _PLACEHOLDER_CODIGO)
        if not self.codigo_aluno_anterior:
            object.__setattr__(self, "codigo_aluno_anterior", self.codigo)
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
            novo_valor = value or (
                _PLACEHOLDER_CODIGO if espelho in ("codigo", "codigo_aluno_anterior")
                else ""
            )
            object.__setattr__(self, espelho, novo_valor)
