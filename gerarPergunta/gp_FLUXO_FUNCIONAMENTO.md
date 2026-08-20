# Fluxo de Funcionamento - Gerador de Perguntas

## Objetivo do fluxo

O fluxo da pasta `gerarPergunta` tem como objetivo transformar respostas ou códigos de estudantes em perguntas personalizadas, adequadas para avaliação formativa e tutoria em programação.

Em termos pedagógicos, o sistema procura criar perguntas que levem o estudante a:

- revisar o próprio código;
- explicar decisões de implementação;
- prever comportamentos de execução;
- identificar erros ou limitações;
- pensar em possíveis modificações.

## Visão geral do processo

```text
1. Recebimento do arquivo de conhecimento
2. Leitura dos exercícios e respostas dos estudantes
3. Criação dos objetos Exercicio
4. Definição dos exercícios que serão usados
5. Sorteio dos tipos de pergunta
6. Montagem do prompt para cada exercício e tipo
7. Envio do prompt ao LLM
8. Recebimento da resposta em JSON
9. Conversão da resposta em objetos Pergunta
10. Exportação para conteudo/perguntasGeradas.txt
```

## 1. Recebimento dos dados do estudante

O processo começa com um arquivo de conhecimento contendo enunciados e respostas ou códigos dos estudantes.

Esse arquivo representa a entrada bruta do módulo. Um exemplo simplificado seria:

```text
1. Contagem de vogais

Resposta 1 -
texto = input("Digite uma frase: ").lower()
...
```

No projeto atual, o arquivo usado como base fica na pasta `conteudo`, enquanto o caminho de entrada é configurado em `config/settings.py`.

## 2. Leitura e análise inicial do conteúdo

A classe `KnowledgeLoader` recebe o caminho do arquivo e faz a leitura linha por linha.

Durante essa etapa, o sistema procura:

- linhas que indicam títulos de questões;
- linhas que indicam início de resposta;
- blocos de código ou texto associados a cada resposta.

O carregador usa expressões regulares para reconhecer esses padrões. A partir disso, separa:

- número da questão;
- título da questão;
- resposta ou código do estudante.

## 3. Criação dos objetos `Exercicio`

Depois da leitura, cada resposta é transformada em um objeto `Exercicio`.

Cada `Exercicio` guarda:

- `numero`: número da questão;
- `titulo`: título ou descrição da questão;
- `codigo`: resposta ou código do estudante.

Se o estudante não escreveu código, o modelo substitui o conteúdo vazio por uma mensagem padrão. Isso evita que as próximas etapas recebam um campo totalmente vazio.

## 4. Definição dos exercícios usados na geração

O `QuestionOrchestrator` recebe a lista de exercícios e a quantidade de perguntas desejada.

Se houver exercícios suficientes, ele usa os primeiros exercícios da lista. Se houver menos exercícios que perguntas, ele repete os exercícios em ciclo.

Exemplo:

```text
2 exercícios disponíveis
5 perguntas desejadas

Slots:
1 -> exercício 1
2 -> exercício 2
3 -> exercício 1
4 -> exercício 2
5 -> exercício 1
```

Essa etapa garante que o sistema consiga gerar a quantidade solicitada mesmo quando o arquivo de entrada tiver poucas respostas.

## 5. Identificação ou escolha do tipo de pergunta

No estado atual do código, o tipo de pergunta não é inferido a partir da resposta do estudante. Ele é sorteado pelo `Sorteador`, com base na lista definida em `TIPOS_PERGUNTA`.

Os tipos disponíveis são:

```text
correcao
justificativa
descritiva
modificacao
previsao
```

O sorteio é feito em blocos embaralhados. Assim, o sistema tenta usar todos os tipos antes de repetir algum deles. Isso favorece variedade pedagógica.

## 6. Montagem do prompt para o LLM

Para cada par formado por exercício e tipo de pergunta, o `AIClient` chama o `QuestionBuilder`.

O `QuestionBuilder` monta um prompt específico. Todos os prompts seguem a mesma estrutura geral:

```text
Papel da IA:
Você é um tutor de programação.

Instrução:
Gere uma pergunta do tipo solicitado.

Contexto:
Exercício, título e código do estudante.

Regras:
- Não fornecer a resposta
- Não copiar o código completo
- Ser claro e objetivo
- Retornar apenas JSON válido
```

Cada tipo de pergunta muda a intenção do prompt:

| Tipo | Intenção do prompt |
| --- | --- |
| `correcao` | Fazer o estudante procurar erro ou caso de borda. |
| `justificativa` | Fazer o estudante explicar uma escolha. |
| `descritiva` | Fazer o estudante descrever o funcionamento do código. |
| `modificacao` | Fazer o estudante pensar em uma alteração. |
| `previsao` | Fazer o estudante prever uma saída. |

Essa etapa é central, porque o prompt controla o comportamento esperado do LLM.

## 7. Envio ao modelo de linguagem

O `AIClient` envia o prompt ao servidor LLM usando a biblioteca `openai`.

Apesar de usar a classe `OpenAI`, a chamada é direcionada para o LM Studio local por meio da configuração:

```text
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
```

A chamada usa:

- uma mensagem de sistema, definindo o modelo como tutor de programação;
- uma mensagem de usuário, contendo o prompt completo;
- o modelo definido em `MODEL`;
- a temperatura definida em `TEMPERATURE`.

## 8. Recebimento da resposta gerada

O LLM deve responder apenas com JSON válido no seguinte formato:

```json
[
  {
    "tipo": "descritiva",
    "pergunta": "Descreva o que o código faz passo a passo."
  }
]
```

O `AIClient` tenta localizar o primeiro array JSON na resposta. Se a resposta vier com texto extra, o sistema ainda tenta extrair o trecho entre `[` e `]`.

Se o JSON for inválido, a pergunta é descartada e o sistema retorna lista vazia para aquela chamada.

## 9. Organização das perguntas geradas

Cada item válido do JSON é convertido em um objeto `Pergunta`.

Esse objeto guarda:

- tipo da pergunta;
- texto da pergunta;
- número do exercício de origem.

O `QuestionOrchestrator` organiza o resultado como uma lista de pares:

```text
(Exercicio, Pergunta)
```

Essa estrutura é importante porque mantém a pergunta ligada ao código que serviu de base para sua geração.

## 10. Saída final do módulo

Depois que as perguntas são geradas, o `ReportExporter` monta o arquivo final.

A saída padrão é:

```text
../conteudo/perguntasGeradas.txt
```

Cada bloco do relatório contém:

- separador visual;
- exercício de origem;
- tipo da pergunta;
- texto da pergunta;
- código do estudante.

Exemplo de estrutura:

```text
============================================================
Exercício gerado com base na sua resposta da questão 1: Questão 1
============================================================

1 - [JUSTIFICATIVA] Por que você converteu a entrada para minúsculas?

Seu código:
----------------------------------------
...
----------------------------------------
```

## Relação com a correção

O arquivo `perguntasGeradas.txt` funciona como ponte entre geração e avaliação.

Em uma execução formativa completa:

1. `gerarPergunta` produz perguntas personalizadas;
2. o estudante responde a essas perguntas;
3. as respostas são registradas no arquivo;
4. `correcao` lê esse arquivo;
5. `correcao` avalia as respostas e gera feedback.

## Fluxo resumido em diagrama

```text
Resposta/código do estudante
        |
        v
KnowledgeLoader
        |
        v
Objeto Exercicio
        |
        v
QuestionOrchestrator
        |
        +--> Sorteador escolhe o tipo
        |
        +--> QuestionBuilder monta o prompt
        |
        +--> AIClient chama o LLM
        |
        v
Objeto Pergunta
        |
        v
ReportExporter
        |
        v
conteudo/perguntasGeradas.txt
```

## Pontos de atenção

- O LM Studio precisa estar aberto e com a API local ativa.
- O modelo deve responder em JSON para que o parser funcione corretamente.
- O caminho do arquivo de entrada deve estar compatível com o diretório de execução.
- A qualidade das perguntas depende fortemente do prompt e do modelo usado.
- O módulo gera perguntas, mas não corrige respostas; essa responsabilidade pertence à pasta `correcao`.

## Contribuição do fluxo para o TCC

Esse fluxo mostra como um LLM pode atuar como tutor formativo em programação. O sistema usa o código real do estudante como contexto e gera perguntas que estimulam metacognição, depuração, explicação e previsão de comportamento.

Assim, a geração de perguntas não é apenas uma etapa técnica, mas uma estratégia pedagógica: ela transforma a produção do estudante em oportunidade de reflexão orientada.
