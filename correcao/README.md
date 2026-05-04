# Sistema de Correção Automática de Questões

Sistema híbrido de correção de questões de programação, combinando execução de código,
casos de teste e avaliação semântica via LLM.

## Estrutura de diretórios

```
correcao/
├── main.py                     # Ponto de entrada
├── config.py                   # Configuração via variáveis de ambiente
│
├── models/
│   └── questao.py              # Dataclasses: Questao e Resultado
│
├── utils/
│   ├── text.py                 # Normalização, comparação e tokenização de texto
│   └── tipo.py                 # Normalização e inferência do tipo de questão
│
├── parsing/
│   └── parser.py               # Leitura e parsing do arquivo de questões (JSON e texto)
│
├── llm/
│   └── client.py               # Cliente HTTP para o servidor LLM (OpenAI-compatible)
│
├── execution/
│   └── runner.py               # Execução isolada de código Python com timeout
│
├── tests/
│   └── generator.py            # Geração, validação e deduplicação de casos de teste
│
├── evaluation/
│   ├── correctors.py           # Avaliadores por tipo (previsao, texto, código, modificação)
│   └── dispatcher.py           # Roteador: escolhe o avaliador certo por tipo de questão
    └── strategies/fa
        ├── __init__.py
        ├── previsao.py                ← executa código-base e compara saída
        ├── correcao.py                ← código vs testes ou texto via LLM
        ├── modificacao.py             ← testes (70%) + requisitos LLM (30%)
        ├── justificativa.py           ← delega para texto_llm
        ├── descritiva.py              ← delega para texto_llm
        ├── texto_llm.py               ← base compartilhada para respostas textuais
        └── codigo.py                  ← base compartilhada para execução com testes
│
└── report/
    └── formatter.py            # Formatação e geração do relatório final em texto
```

## Tipos de questão suportados

| Tipo           | Estratégia de correção                                      |
|----------------|-------------------------------------------------------------|
| `previsao`     | Executa o código-base e compara a saída com a resposta      |
| `correcao`     | Testa o código do aluno ou avalia textualmente via LLM      |
| `modificacao`  | Testes (70%) + checagem de requisitos via LLM (30%)         |
| `justificativa`| Avaliação semântica via LLM                                 |
| `descritiva`   | Avaliação semântica via LLM                                 |

## Configuração

Copie e edite as variáveis conforme necessário:

```bash
export ARQUIVO_ENTRADA="conteudo/perguntasGeradas.txt"
export ARQUIVO_SAIDA="conteudo/correcao.txt"
export LLM_BASE_URL="http://localhost:1234/v1"
export LLM_MODEL="qwen/qwen3-vl-4b"
export USAR_LLM="1"          # "0" para desabilitar o LLM
export LLM_TIMEOUT="90"
```

## Execução

```bash
cd correcao/
python main.py
```

## Formato de entrada

### JSON
```json
[
  {
    "id": 1,
    "tipo": "correcao",
    "enunciado": "Corrija o código abaixo...",
    "codigo": "def soma(a, b):\n    return a - b",
    "resposta_aluno": "def soma(a, b):\n    return a + b",
    "testes": [
      {"entrada": "2\n3\n", "saida": "5\n"}
    ]
  }
]
```

### Texto em blocos
```
========== Exercicio gerado com base na sua resposta da questão 1 ==========
1 - [CORRECAO] Corrija o erro no código abaixo
Código:
def soma(a, b):
    return a - b
Resposta 1 - def soma(a, b):
    return a + b
```
