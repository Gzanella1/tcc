# =============================================================
# services/report_exporter.py
# Responsabilidade única: formatar e salvar o relatório final
# com as perguntas geradas.
# =============================================================

from pathlib import Path

from models.exercicio import Exercicio
from models.pergunta   import Pergunta
from config.settings   import ARQUIVO_SAIDA


class ReportExporter:
    """
    Formata os resultados gerados e os salva em um arquivo .txt.

    Separar a formatação do relatório facilita trocar o formato
    de saída no futuro (HTML, PDF, JSON, etc.) sem mexer no resto.
    """

    def __init__(self, caminho_saida: str = ARQUIVO_SAIDA):
        self.caminho = Path(caminho_saida)

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def exportar(self, pares: list[tuple[Exercicio, Pergunta]]) -> None:
        """
        Recebe a lista de (Exercicio, Pergunta) e grava o arquivo.

        Parâmetros:
            pares – lista de tuplas (Exercicio, Pergunta)
        """
        conteudo = self._formatar(pares)
        self._salvar(conteudo)

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _formatar(self, pares: list[tuple[Exercicio, Pergunta]]) -> str:
        """Monta o texto completo do relatório."""
        linhas = []

        for idx, (exercicio, pergunta) in enumerate(pares, start=1):
            separador = f"{'='*60}"
            linhas.append(separador)
            linhas.append(
                f"Exercício gerado com base na sua resposta da questão "
                f"{exercicio.numero}: {exercicio.titulo}"
            )
            linhas.append(separador)
            linhas.append("")
            linhas.append(
                f"{idx} - [{pergunta.tipo.upper()}] {pergunta.pergunta}"
            )
            linhas.append("")
            linhas.append("Seu código:")
            linhas.append("-" * 40)
            linhas.append(exercicio.codigo)
            linhas.append("-" * 40)
            linhas.append("")
            linhas.append("")   # espaço extra entre blocos

        return "\n".join(linhas)

    def _salvar(self, conteudo: str) -> None:
        """Cria o diretório (se necessário) e escreve o arquivo."""
        self.caminho.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.caminho.write_text(conteudo, encoding="utf-8")
            print(f"\n✅ Perguntas salvas em '{self.caminho}'")
        except OSError as exc:
            print(f"\n❌ Erro ao salvar arquivo: {exc}")
            raise
