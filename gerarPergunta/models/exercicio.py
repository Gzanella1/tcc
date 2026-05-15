# =============================================================
# models/exercicio.py
# Modelo de dados que representa um exercício lido do arquivo
# =============================================================

from dataclasses import dataclass, field


@dataclass
class Exercicio:
    """
    Representa um único exercício com o código enviado pelo aluno.

    Atributos:
        numero  – número da questão no arquivo de conhecimento
        titulo  – enunciado resumido da questão
        codigo  – resposta/código escrito pelo aluno
    """
    numero: int
    titulo: str
    codigo: str = field(default="[O ALUNO NÃO ESCREVEU CÓDIGO]")

    def __post_init__(self):
        # Garante que um código vazio seja substituído pelo texto padrão
        if not self.codigo or not self.codigo.strip():
            self.codigo = "[O ALUNO NÃO ESCREVEU CÓDIGO]"

    def __repr__(self):
        preview = self.codigo[:60].replace("\n", " ")
        return f"Exercicio(numero={self.numero}, titulo='{self.titulo}', codigo='{preview}...')"
