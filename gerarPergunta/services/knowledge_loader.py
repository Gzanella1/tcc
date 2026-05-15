# =============================================================
# services/knowledge_loader.py
# Responsabilidade única: ler e fazer o parse do arquivo .txt
# com as questões e respostas dos alunos.
# =============================================================

import re
from pathlib import Path
from models.exercicio import Exercicio


class KnowledgeLoader:
    """
    Lê o arquivo de conhecimento no formato:

        1. Título da questão
        2. Outro título

        Resposta 1 -
        <código do aluno>

        Resposta 2 -
        <código do aluno>

    e retorna uma lista de objetos Exercicio.
    """

    # Regex compiladas uma única vez (melhor performance)
    _RE_QUESTAO   = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*$")
    _RE_RESPOSTA  = re.compile(r"^\s*Resposta\s*(\d+)\s*-\s*$", re.IGNORECASE)

    def __init__(self, caminho_arquivo: str):
        self.caminho = Path(caminho_arquivo)

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def carregar(self) -> list[Exercicio]:
        """Lê o arquivo e retorna a lista de Exercicio."""
        if not self.caminho.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {self.caminho}")

        linhas = self.caminho.read_text(encoding="utf-8").splitlines(keepends=True)
        titulos, respostas = self._parsear(linhas)

        return self._montar_exercicios(titulos, respostas)

    # ------------------------------------------------------------------
    # Métodos privados de parsing
    # ------------------------------------------------------------------

    def _parsear(self, linhas: list[str]) -> tuple[dict, dict]:
        """
        Percorre as linhas e separa títulos e blocos de código.

        Retorna:
            titulos   – {numero: "título da questão"}
            respostas – {numero: "código do aluno"}
        """
        titulos:  dict[int, str]  = {}
        respostas: dict[int, str] = {}

        numero_atual: int | None  = None
        linhas_codigo: list[str]  = []

        def _salvar_resposta_atual():
            nonlocal numero_atual, linhas_codigo
            if numero_atual is not None:
                respostas[numero_atual] = "\n".join(linhas_codigo).strip()
            numero_atual = None
            linhas_codigo = []

        for raw in linhas:
            linha = raw.rstrip("\n")
            stripped = linha.strip()

            m_questao  = self._RE_QUESTAO.match(stripped)
            m_resposta = self._RE_RESPOSTA.match(stripped)

            if m_questao:
                titulos[int(m_questao.group(1))] = m_questao.group(2)
                continue

            if m_resposta:
                _salvar_resposta_atual()          # fecha o bloco anterior
                numero_atual = int(m_resposta.group(1))
                continue

            if numero_atual is not None:          # dentro de um bloco de resposta
                linhas_codigo.append(linha)

        _salvar_resposta_atual()                  # fecha o último bloco
        return titulos, respostas

    def _montar_exercicios(
        self,
        titulos: dict[int, str],
        respostas: dict[int, str],
    ) -> list[Exercicio]:
        """Combina títulos e respostas em objetos Exercicio ordenados."""
        if not respostas:
            raise ValueError("Nenhuma resposta válida encontrada no arquivo.")

        return [
            Exercicio(
                numero=num,
                titulo=titulos.get(num, f"Questão {num}"),
                codigo=respostas[num],
            )
            for num in sorted(respostas)
        ]
