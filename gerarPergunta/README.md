# TCC — Tutor de Programação com IA

## Estrutura do Projeto

```
tcc_project/
│
├── main.py                         # Ponto de entrada — só orquestra, zero lógica
│
├── conhecimento.txt                # Arquivo com questões e respostas do aluno
│
├── config/
│   ├── __init__.py
│   └── settings.py                 # Todas as configurações em um lugar só
│
├── models/                         # Modelos de dados (sem lógica de negócio)
│   ├── __init__.py
│   ├── exercicio.py                # Dataclass Exercicio
│   └── pergunta.py                 # Dataclass Pergunta
│
├── services/                       # Lógica de negócio — cada arquivo = 1 responsabilidade
│   ├── __init__.py
│   ├── knowledge_loader.py         # Lê e faz parse do arquivo .txt
│   ├── question_builder.py         # Monta o prompt certo para cada tipo de pergunta
│   ├── ai_client.py                # Fala com a API do LM Studio e retorna Perguntas
│   ├── question_orchestrator.py    # Coordena o fluxo: slots → tipos → IA
│   └── report_exporter.py          # Formata e salva o relatório final
│
├── utils/
│   ├── __init__.py
│   └── sorteador.py                # Sorteia tipos de pergunta sem repetir no bloco
│
└── conteudo/
    └── perguntasGeradas.txt        # Saída gerada pelo sistema
```

## Responsabilidade de cada arquivo

| Arquivo | Responsabilidade |
|---|---|
| `config/settings.py` | Centraliza URL, model, total de perguntas, caminhos |
| `models/exercicio.py` | Estrutura de dados de um exercício (número, título, código) |
| `models/pergunta.py` | Estrutura de dados de uma pergunta gerada (tipo, texto) |
| `services/knowledge_loader.py` | Lê o `conhecimento.txt` e retorna lista de `Exercicio` |
| `services/question_builder.py` | Gera o prompt certo para cada um dos 5 tipos de pergunta |
| `services/ai_client.py` | Chama a API, extrai o JSON, converte em `Pergunta` |
| `services/question_orchestrator.py` | Decide quais exercícios usar, sorteia tipos, chama a IA |
| `services/report_exporter.py` | Formata e salva o `.txt` final |
| `utils/sorteador.py` | Embaralha tipos garantindo cobertura uniforme |
| `main.py` | Liga tudo: carrega → gera → exporta |

## Como adicionar um novo tipo de pergunta

1. Adicione o nome em `config/settings.py` → `TIPOS_PERGUNTA`
2. Adicione um método em `services/question_builder.py` decorado com `@_registrar("novo_tipo")`

Só isso. O resto do sistema já vai incluir o novo tipo automaticamente.

## Como executar

```bash
pip install openai
python main.py
```
