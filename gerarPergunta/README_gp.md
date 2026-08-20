# Gerador de Perguntas

## Objetivo da pasta

A pasta `gerarPergunta` implementa o módulo responsável por gerar perguntas personalizadas a partir das respostas ou códigos escritos por estudantes. No contexto do TCC, esse módulo representa a etapa de tutoria formativa: em vez de apenas informar se a resposta está certa ou errada, o sistema produz perguntas que estimulam o estudante a refletir sobre o próprio código.

Embora o usuário possa se referir a esta parte como `gerar_pergunta`, no repositório a pasta está nomeada como `gerarPergunta`.

## Responsabilidade no sistema

Esta pasta recebe exercícios e respostas de estudantes, identifica quais perguntas devem ser produzidas, monta prompts específicos para um Grande Modelo de Linguagem (LLM) e salva as perguntas geradas em um arquivo de saída. Ela funciona como uma etapa anterior à pasta `correcao`: primeiro as perguntas são geradas, depois as respostas dos estudantes a essas perguntas podem ser avaliadas pelo módulo de correção.

## Arquivos principais

| Arquivo | Função |
| --- | --- |
| `main.py` | Ponto de entrada do módulo. Coordena o carregamento dos exercícios, a geração das perguntas e a exportação do resultado. |
| `config/settings.py` | Centraliza configurações como URL do LM Studio, modelo usado, temperatura, total de perguntas, tipos de pergunta e caminhos de entrada/saída. |
| `models/exercicio.py` | Define a classe `Exercicio`, que representa uma questão com número, título e código/resposta do estudante. |
| `models/pergunta.py` | Define a classe `Pergunta`, que representa uma pergunta gerada pelo LLM, contendo tipo, texto e número do exercício de origem. |
| `services/knowledge_loader.py` | Lê o arquivo de conhecimento e transforma blocos de resposta em objetos `Exercicio`. |
| `services/question_builder.py` | Monta o prompt adequado para cada tipo de pergunta. |
| `services/ai_client.py` | Comunica-se com o servidor LLM compatível com a API da OpenAI e converte a resposta JSON em objetos `Pergunta`. |
| `services/question_orchestrator.py` | Coordena a combinação entre exercícios, tipos de pergunta e chamadas ao LLM. |
| `services/report_exporter.py` | Formata e salva as perguntas geradas no arquivo final. |
| `utils/sorteador.py` | Sorteia tipos de pergunta, garantindo variedade antes de repetir categorias. |

## Entradas

A entrada principal é um arquivo de texto contendo enunciados e respostas/códigos de estudantes. O carregador espera blocos no seguinte formato geral:

```text
1. Título da questão

Resposta 1 -
código ou resposta do estudante
```

O caminho de entrada é definido em `config/settings.py` pela constante `ARQUIVO_CONHECIMENTO`. No repositório atual, o arquivo de conhecimento está em `../conteudo/conhecimento.txt` quando a execução ocorre de dentro da pasta `gerarPergunta`.

## Saídas

A saída principal é um arquivo de texto com as perguntas geradas, por padrão:

```text
../conteudo/perguntasGeradas.txt
```

Esse arquivo contém blocos com:

- identificação do exercício de origem;
- tipo da pergunta;
- pergunta gerada;
- código do estudante usado como base.

Esse resultado pode ser usado posteriormente pelo módulo `correcao`, especialmente quando as respostas dos estudantes forem adicionadas ao arquivo.

## Tipos de pergunta

O módulo trabalha com cinco categorias pedagógicas:

| Tipo | Objetivo formativo |
| --- | --- |
| `correcao` | Levar o estudante a identificar erros, falhas lógicas ou casos de borda. |
| `justificativa` | Solicitar que o estudante explique uma decisão de implementação. |
| `descritiva` | Pedir uma descrição do funcionamento do código. |
| `modificacao` | Propor uma alteração, melhoria ou extensão do código. |
| `previsao` | Pedir que o estudante preveja a saída do programa para determinada entrada. |

## Tecnologias e recursos utilizados

- Python 3;
- `dataclasses`, para representar exercícios e perguntas;
- `pathlib`, para manipulação de caminhos de arquivos;
- `re`, para leitura estruturada do arquivo de conhecimento;
- `random`, para sorteio dos tipos de pergunta;
- `json`, para interpretar a resposta do LLM;
- `functools.lru_cache`, para reaproveitar chamadas repetidas durante a execução;
- biblioteca `openai`, configurada para conversar com um servidor local do LM Studio;
- LM Studio, expondo uma API compatível com OpenAI em `http://localhost:1234/v1`;
- modelo configurado em `MODEL`, atualmente `qwen/qwen3-vl-4b`.

## Relação com o restante do projeto

O fluxo geral do projeto pode ser entendido assim:

```text
conteudo/conhecimento.txt
        |
        v
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

Assim, `gerarPergunta` prepara perguntas personalizadas a partir da produção do estudante, enquanto `correcao` avalia as respostas posteriormente registradas. As duas pastas se complementam: uma atua na geração de intervenções formativas, e a outra na avaliação automatizada dessas respostas.

## Contribuição para o TCC

Esta pasta contribui diretamente para o objetivo geral do TCC ao demonstrar como LLMs podem ser usados para apoiar a avaliação formativa no ensino de programação. O módulo não se limita a gerar exercícios genéricos: ele usa o código real do estudante como contexto para produzir perguntas mais personalizadas, favorecendo reflexão, justificativa, previsão de comportamento, revisão de erros e adaptação de soluções.

Com isso, o sistema se aproxima de uma tutoria automatizada, em que o feedback não é apenas classificatório, mas também orientado ao desenvolvimento do raciocínio computacional.
