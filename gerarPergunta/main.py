# =============================================================
# main.py
# Ponto de entrada do sistema — apenas orquestra os serviços.
# Nenhuma lógica de negócio aqui.
# =============================================================

from config.settings            import ARQUIVO_CONHECIMENTO, ARQUIVO_SAIDA, TOTAL_PERGUNTAS
from services.knowledge_loader  import KnowledgeLoader
from services.question_orchestrator import QuestionOrchestrator
from services.report_exporter   import ReportExporter


def main():
    print("=" * 60)
    print("  TCC — Tutor de Programação com IA")
    print("=" * 60)

    # 1. Carrega exercícios do arquivo
    print(f"\n📂 Carregando exercícios de '{ARQUIVO_CONHECIMENTO}'...")
    loader     = KnowledgeLoader(ARQUIVO_CONHECIMENTO)
    exercicios = loader.carregar()
    print(f"   {len(exercicios)} exercício(s) carregado(s).")

    # 2. Gera as perguntas via IA
    print(f"\n🤖 Gerando {TOTAL_PERGUNTAS} pergunta(s)...\n")
    orchestrator = QuestionOrchestrator()
    pares        = orchestrator.gerar(exercicios, total=TOTAL_PERGUNTAS)

    # 3. Exporta o relatório
    print(f"\n💾 Salvando relatório em '{ARQUIVO_SAIDA}'...")
    exporter = ReportExporter(ARQUIVO_SAIDA)
    exporter.exportar(pares)

    print("\n🎉 Concluído!\n")


if __name__ == "__main__":
    main()
