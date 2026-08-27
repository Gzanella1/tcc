# Sistema de Correção

## Objetivo da pasta

A pasta `correcao` implementa o módulo responsável por corrigir automaticamente respostas de estudantes a questões de programação. Ela combina estratégias objetivas, como execução de código e comparação de saídas, com estratégias semânticas apoiadas por um Grande Modelo de Linguagem (LLM).

No contexto do TCC, esta pasta representa a etapa de avaliação formativa: além de atribuir nota ou status, o sistema produz feedback textual, registra detalhes da correção e indica pontos de acerto ou melhoria.

## Responsabilidade no sistema

O módulo recebe questões e respostas de estudantes, identifica o tipo de questão, escolhe a estratégia de correção mais adequada e gera um relatório final. A correção pode envolver:

- execução de código Python em processo separado;
- comparação entre saída esperada e saída obtida;
- geração ou uso de casos de teste;
- avaliação textual com apoio de LLM;
- combinação ponderada entre testes automatizados e análise semântica.

## Arquivos principais

| Arquivo | Função |
| --- | --- |
| `main.py` | Ponto de entrada. Lê as questões, corrige cada uma e salva o relatório final. |
| `config.py` | Centraliza caminhos, configurações do LLM, limiares de similaridade e quantidade-alvo de testes. |
| `models/questao.py` | Define os modelos `Questao` e `Resultado`, usados em todo o fluxo de correção. |
| `evaluation/evidencia.py` | Fonte única de verdade da rastreabilidade: constantes de fonte (`execucao`, `llm`, `heuristica`, `ausente`) e tipo de evidência, além do contrato `{tipo, resumo, dados[, peso]}` e da normalização da origem. |
| `parsing/parser.py` | Lê o arquivo de entrada e converte JSON ou texto em blocos para objetos `Questao`. |
| `evaluation/dispatcher.py` | Roteia cada questão para a estratégia correta conforme o tipo identificado. |
| `evaluation/correctors.py` | Arquivo com implementações de avaliadores em formato mais concentrado; funciona como referência/versão anterior em relação à arquitetura por estratégias. |
| `evaluation/strategies/codigo.py` | Avalia código Python por execução e casos de teste, registrando evidências estruturadas (incluindo `testes_por_origem`, a procedência da régua de testes). |
| `evaluation/strategies/previsao.py` | Corrige questões em que o estudante prevê a saída de um programa. |
| `evaluation/strategies/correcao.py` | Corrige questões de correção de erro, escolhendo entre avaliação textual e execução de código. |
| `evaluation/strategies/modificacao.py` | Corrige modificações de código usando testes e análise de requisitos via LLM. |
| `evaluation/strategies/texto_llm.py` | Avalia respostas textuais com apoio do LLM. |
| `evaluation/strategies/descritiva.py` | Reaproveita a avaliação textual para questões descritivas. |
| `evaluation/strategies/justificativa.py` | Reaproveita a avaliação textual para questões de justificativa. |
| `execution/runner.py` | Executa código Python em subprocesso isolado, com timeout e captura de saída. |
| `tests/generator.py` | Obtém, valida, deduplica e, quando necessário, gera testes com apoio do LLM, marcando a procedência interna de cada teste (`_origem`: `enunciado` ou `llm`). |
| `llm/client.py` | Realiza chamadas HTTP para um servidor LLM compatível com a API da OpenAI. |
| `utils/text.py` | Reúne funções de normalização, extração de código/JSON e comparação textual. |
| `utils/tipo.py` | Normaliza e infere o tipo da questão. |
| `report/formatter.py` | Formata o relatório final de correção em texto simples. |

## Entradas

A entrada padrão é definida em `config.py`:

```text
../conteudo/perguntasGeradas.txt
```

Esse arquivo pode conter perguntas geradas pelo módulo `gerarPergunta` e respostas adicionadas posteriormente pelo estudante. O parser também aceita entrada em JSON, desde que contenha campos como `tipo`, `enunciado`, `resposta_aluno`, `codigo`, `entrada`, `saida` ou `testes`.

## Saídas

A saída padrão é:

```text
../conteudo/correcao.txt
```

O relatório produzido contém:

- data da correção;
- total de questões avaliadas;
- média geral;
- quantidade de questões com nota igual ou superior a 7;
- nota individual;
- status da resposta;
- feedback;
- enunciado;
- detalhes da avaliação;
- testes executados, quando houver;
- evidências estruturadas da correção, quando houver (seção "Evidências:", com tipo, resumo e dados de cada evidência, incluindo a procedência dos testes usados como régua);
- saídas esperadas e obtidas.

## Tipos de questão suportados

| Tipo | Estratégia de correção |
| --- | --- |
| `previsao` | Executa o código-base e compara a saída real com a previsão do estudante. |
| `correcao` | Se a resposta for textual, usa LLM; se for código e houver testes, executa e compara resultados. |
| `modificacao` | Combina execução com testes, com peso de 70%, e análise de requisitos via LLM, com peso de 30%. |
| `justificativa` | Usa avaliação textual via LLM, priorizando entendimento conceitual. |
| `descritiva` | Usa avaliação textual via LLM, priorizando clareza e completude da explicação. |

## Tecnologias e recursos utilizados

- Python 3;
- `dataclasses`, para os modelos de dados;
- `pathlib`, para caminhos de arquivos;
- `json` e `re`, para parsing de entrada e respostas do LLM;
- `ast`, para validação sintática de código Python;
- `subprocess`, `sys` e `tempfile`, para execução isolada de código;
- `difflib.SequenceMatcher`, para cálculo de similaridade textual;
- `urllib.request`, para chamadas HTTP ao servidor LLM;
- LM Studio ou outro servidor local compatível com a API OpenAI;
- modelo configurado em `LLM_MODEL`, atualmente `qwen/qwen3-vl-4b`.

## Relação com o restante do projeto

A pasta `correcao` recebe como entrada o arquivo produzido pela pasta `gerarPergunta`, depois que as respostas dos estudantes são incluídas. Assim, o fluxo geral é:

```text
gerarPergunta
        |
        v
conteudo/perguntasGeradas.txt
        |
        v
correcao
        |
        v
conteudo/correcao.txt
```

Essa separação deixa o projeto mais organizado: a geração das perguntas fica concentrada em um módulo, enquanto a avaliação das respostas fica concentrada em outro.

## Contribuição para o TCC

Este módulo contribui para o objetivo geral do TCC ao demonstrar uma abordagem híbrida para avaliação formativa em programação. O sistema não depende apenas de gabaritos fixos nem apenas de respostas abertas avaliadas por LLM. Em vez disso, combina execução automatizada, testes, comparação textual e análise semântica.

Essa combinação permite avaliar diferentes dimensões da aprendizagem: funcionamento do código, previsão de execução, capacidade de explicar decisões, compreensão do comportamento do programa e atendimento a requisitos de modificação. Com isso, o projeto oferece uma base prática para discutir o uso de LLMs como apoio à tutoria e à avaliação formativa no ensino de programação.
