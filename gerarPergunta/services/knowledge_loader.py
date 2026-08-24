# =============================================================
# services/knowledge_loader.py
# Responsabilidade única: ler e fazer o parse do arquivo .txt
# com as questões e respostas dos alunos.
#
# Fase 3.1: o enunciado original de cada questão é preservado
# separadamente do código produzido pelo aluno. Formatos de
# enunciado aceitos: "N. Título" (legado) e "N - Enunciado"
# (formato real do conhecimento.txt). Linhas de enunciado só são
# reconhecidas FORA de blocos "resposta", para nunca capturar
# código do aluno como título.
# =============================================================

import re
from pathlib import Path
from models.exercicio import Exercicio


class KnowledgeLoader:
    """
    Lê o arquivo de conhecimento no formato:

        1 - Enunciado original da questão 1
        2 - Enunciado original da questão 2

        resposta 1 -
        <código produzido anteriormente pelo aluno>

        resposta 2 -
        <código produzido anteriormente pelo aluno>

    e retorna uma lista de objetos Exercicio contendo, separadamente,
    o enunciado original e a resposta/código anterior do aluno.
    """

    # Regex compiladas uma única vez (melhor performance)
    _RE_ENUNCIADO = re.compile(r"^\s*(\d+)(?:\.\s*|\s+-\s+)(.+?)\s*$")
    _RE_RESPOSTA = re.compile(r"^\s*Resposta\s*(\d+)\s*-\s*$", re.IGNORECASE)

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
        enunciados, respostas = self._parsear(linhas)

        return self._montar_exercicios(enunciados, respostas)

    # ------------------------------------------------------------------
    # Métodos privados de parsing
    # ------------------------------------------------------------------

    def _parsear(self, linhas: list[str]) -> tuple[dict, dict]:
        """
        Percorre as linhas e separa enunciados originais e blocos de código.

        Retorna:
            enunciados – {numero: "enunciado original da questão"}
            respostas  – {numero: "código produzido anteriormente pelo aluno"}
        """
        enunciados: dict[int, str] = {}
        respostas: dict[int, str] = {}

        numero_atual: int | None = None
        linhas_codigo: list[str] = []

        def _salvar_resposta_atual():
            nonlocal numero_atual, linhas_codigo
            if numero_atual is not None:
                respostas[numero_atual] = "\n".join(linhas_codigo).strip()
            numero_atual = None
            linhas_codigo = []

        for raw in linhas:
            linha = raw.rstrip("\n")
            stripped = linha.strip()

            m_resposta = self._RE_RESPOSTA.match(stripped)

            if m_resposta:
                _salvar_resposta_atual()          # fecha o bloco anterior
                numero_atual = int(m_resposta.group(1))
                continue

            if numero_atual is not None:          # dentro de um bloco de resposta
                linhas_codigo.append(linha)
                continue

            # Fora de bloco de resposta: captura o enunciado original.
            # Assim, linhas tipo "1 - Soma" dentro de código nunca viram título.
            m_enunciado = self._RE_ENUNCIADO.match(stripped)
            if m_enunciado:
                enunciados[int(m_enunciado.group(1))] = m_enunciado.group(2)

        _salvar_resposta_atual()                  # fecha o último bloco
        return enunciados, respostas

    def _montar_exercicios(
        self,
        enunciados: dict[int, str],
        respostas: dict[int, str],
    ) -> list[Exercicio]:
        """Combina enunciados e respostas em objetos Exercicio ordenados."""
        if not respostas:
            raise ValueError("Nenhuma resposta válida encontrada no arquivo.")

        return [
            Exercicio(
                numero=num,
                titulo=enunciados.get(num, ""),
                codigo=respostas[num],
                enunciado_original=enunciados.get(num, ""),
            )
            for num in sorted(respostas)
        ]
