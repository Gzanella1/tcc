# =============================================================
# config/settings.py
# Configurações centrais do projeto
# =============================================================

# --- LM Studio / OpenAI API ---
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY  = "lm-studio"
MODEL              = "qwen/qwen3-vl-4b"

# --- Comportamento da geração ---
TOTAL_PERGUNTAS   = 5
TEMPERATURE       = 0.4
MAX_TOKENS        = 512          # opcional; remova se não quiser limitar

# --- Tipos de pergunta disponíveis ---
TIPOS_PERGUNTA = [
    "correcao",
    "justificativa",
    "descritiva",
    "modificacao",
    "previsao",
]

# --- Caminhos de arquivos ---
ARQUIVO_CONHECIMENTO  = "../conteudo/conhecimento.txt"
ARQUIVO_SAIDA         = "../conteudo/perguntasGeradas.txt"
