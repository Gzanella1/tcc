# Explicação do Código - Gerador de Perguntas

## Visão geral

O módulo `gerarPergunta` transforma respostas ou códigos de estudantes em perguntas personalizadas geradas por um LLM. A organização do código segue uma divisão por responsabilidades:

- `main.py` inicia e coordena o processo;
- `config/` guarda configurações;
- `models/` define as estruturas de dados;
- `services/` concentra a lógica principal;
- `utils/` guarda funções auxiliares.

O fluxo central é:

```text
arquivo de conhecimento -> Exercicio -> prompt -> LLM -> Pergunta -> relatório
```

## `main.py`

### Função do arquivo

É o ponto de entrada do módulo. Ele não contém a lógica detalhada de geração; sua função é orquestrar os serviços.

### Função principal

#### `main()`

Recebe:

- não recebe parâmetros diretamente;
- usa as constantes `ARQUIVO_CONHECIMENTO`, `ARQUIVO_SAIDA` e `TOTAL_PERGUNTAS`, importadas de `config/settings.py`.

Faz:

1. cria um `KnowledgeLoader`;
2. carrega os exercícios do arquivo de conhecimento;
3. cria um `QuestionOrchestrator`;
4. solicita a geração das perguntas;
5. cria um `ReportExporter`;
6. salva o resultado final.

Retorna:

- não retorna valor útil; apenas executa o fluxo e imprime mensagens no terminal.

### Conexões

`main.py` conecta os três serviços principais:

- `KnowledgeLoader`, para entrada;
- `QuestionOrchestrator`, para geração;
- `ReportExporter`, para saída.

É o melhor arquivo para explicar ao professor a visão geral da execução.

## `config/settings.py`

### Função do arquivo

Centraliza os parâmetros usados pelo gerador de perguntas.

### Configurações principais

| Constante | Função |
| --- | --- |
| `LM_STUDIO_BASE_URL` | Endereço da API local do LM Studio. |
| `LM_STUDIO_API_KEY` | Chave usada pelo cliente OpenAI. No LM Studio local, pode ser um valor simbólico. |
| `MODEL` | Nome do modelo usado para gerar perguntas. |
| `TOTAL_PERGUNTAS` | Quantidade de perguntas que o sistema tenta gerar. |
| `TEMPERATURE` | Controla a criatividade do LLM. |
| `MAX_TOKENS` | Limite previsto para resposta, embora atualmente não seja passado na chamada da API. |
| `TIPOS_PERGUNTA` | Lista dos tipos pedagógicos disponíveis. |
| `ARQUIVO_CONHECIMENTO` | Caminho do arquivo de entrada. |
| `ARQUIVO_SAIDA` | Caminho do arquivo onde as perguntas serão salvas. |

### Conexões

Este arquivo é importado por vários módulos:

- `main.py`, para saber caminhos e quantidade de perguntas;
- `AIClient`, para configurar o LLM;
- `Sorteador`, para conhecer os tipos de pergunta;
- `ReportExporter`, para o caminho padrão de saída.

## `models/exercicio.py`

### Função do arquivo

Define o modelo de dados `Exercicio`, que representa uma questão original e o código/resposta do estudante.

### Classe principal

#### `Exercicio`

Atributos:

- `numero`: número da questão;
- `titulo`: título ou descrição curta da questão;
- `codigo`: código ou resposta do estudante.

### Métodos

#### `__post_init__()`

Recebe:

- os próprios dados do objeto após sua criação.

Faz:

- verifica se `codigo` está vazio;
- se estiver, substitui por `[O ALUNO NÃO ESCREVEU CÓDIGO]`.

Retorna:

- não retorna valor; apenas ajusta o atributo `codigo`.

#### `__repr__()`

Faz:

- cria uma representação curta do exercício para depuração.

Retorna:

- uma string com número, título e prévia do código.

### Conexões

`Exercicio` é criado por `KnowledgeLoader` e usado por `QuestionBuilder`, `AIClient`, `QuestionOrchestrator` e `ReportExporter`.

## `models/pergunta.py`

### Função do arquivo

Define o modelo de dados `Pergunta`, que representa uma pergunta gerada pelo LLM.

### Classe principal

#### `Pergunta`

Atributos:

- `tipo`: categoria da pergunta, como `correcao`, `previsao` ou `descritiva`;
- `pergunta`: texto da pergunta;
- `exercicio_numero`: número do exercício que originou a pergunta.

### Método

#### `__str__()`

Faz:

- formata a pergunta como texto legível.

Retorna:

- uma string no formato `[TIPO] pergunta`.

### Conexões

Objetos `Pergunta` são criados em `AIClient` após o parsing da resposta do LLM e depois enviados ao `ReportExporter`.

## `services/knowledge_loader.py`

### Função do arquivo

Lê o arquivo de conhecimento e transforma seu conteúdo em uma lista de objetos `Exercicio`.

### Classe principal

#### `KnowledgeLoader`

Recebe na criação:

- `caminho_arquivo`: caminho do arquivo de entrada.

Guarda:

- `self.caminho`, como objeto `Path`.

### Métodos principais

#### `carregar()`

Recebe:

- nenhum parâmetro além do próprio objeto.

Faz:

1. verifica se o arquivo existe;
2. lê as linhas em UTF-8;
3. chama `_parsear()` para separar títulos e respostas;
4. chama `_montar_exercicios()` para criar objetos `Exercicio`.

Retorna:

- `list[Exercicio]`.

#### `_parsear(linhas)`

Recebe:

- lista de linhas do arquivo.

Faz:

- identifica títulos no formato `1. Título`;
- identifica respostas no formato `Resposta 1 -`;
- associa cada bloco de código ao número correspondente.

Retorna:

- uma tupla `(titulos, respostas)`, em que cada item é um dicionário indexado pelo número da questão.

#### `_montar_exercicios(titulos, respostas)`

Recebe:

- dicionário de títulos;
- dicionário de respostas.

Faz:

- combina cada resposta com seu título;
- usa título genérico quando o título não é encontrado;
- ordena os exercícios pelo número.

Retorna:

- lista de objetos `Exercicio`.

### Partes importantes

As expressões regulares `_RE_QUESTAO` e `_RE_RESPOSTA` são essenciais, pois definem o formato esperado do arquivo. Se o texto de entrada não seguir esse padrão, o carregador pode não identificar corretamente os títulos.

## `utils/sorteador.py`

### Função do arquivo

Controla o sorteio dos tipos de pergunta.

### Classe principal

#### `Sorteador`

Recebe na criação:

- opcionalmente, uma lista de tipos;
- se nenhum tipo for informado, usa `TIPOS_PERGUNTA`.

### Método

#### `tipos_aleatorios(quantidade)`

Recebe:

- `quantidade`: número de tipos que devem ser retornados.

Faz:

- copia a lista de tipos;
- embaralha a cópia;
- adiciona blocos embaralhados até atingir a quantidade desejada;
- evita repetição antes de todos os tipos aparecerem dentro de um bloco.

Retorna:

- `list[str]`, contendo os tipos sorteados.

### Conexões

É usado por `QuestionOrchestrator` para definir que tipo de pergunta será gerado para cada exercício.

## `services/question_builder.py`

### Função do arquivo

Monta o prompt que será enviado ao LLM. Cada tipo de pergunta tem uma instrução própria.

### Classe principal

#### `QuestionBuilder`

Não recebe parâmetros obrigatórios na criação.

### Método público

#### `construir(exercicio, tipo)`

Recebe:

- `exercicio`: objeto `Exercicio`;
- `tipo`: string indicando o tipo de pergunta.

Faz:

- procura o método privado correspondente ao tipo;
- chama esse método para montar o prompt.

Retorna:

- uma string com o prompt completo.

Se o tipo não existir, lança `ValueError`.

### Métodos auxiliares

#### `_cabecalho(exercicio)`

Monta o trecho do prompt com número, título e código do estudante.

#### `_rodape(tipo)`

Adiciona regras obrigatórias, como:

- não fornecer a resposta;
- não copiar o código completo;
- retornar apenas JSON válido.

### Métodos por tipo

| Método | Objetivo |
| --- | --- |
| `_correcao()` | Solicita pergunta sobre erro lógico, sintático ou caso de borda. |
| `_justificativa()` | Solicita pergunta sobre a razão de uma escolha de implementação. |
| `_descritiva()` | Solicita pergunta para descrever o funcionamento do código. |
| `_modificacao()` | Solicita pergunta envolvendo alteração, refatoração ou nova funcionalidade. |
| `_previsao()` | Solicita pergunta sobre a saída esperada para determinada entrada. |

### Conexões

`QuestionBuilder` é usado por `AIClient`. Ele é uma das partes mais importantes do projeto, porque a qualidade da pergunta depende diretamente da qualidade do prompt.

## `services/ai_client.py`

### Função do arquivo

Faz a comunicação com o LLM e transforma a resposta textual em objetos `Pergunta`.

### Classe principal

#### `AIClient`

Na criação:

- instancia `OpenAI` apontando para `LM_STUDIO_BASE_URL`;
- cria um `QuestionBuilder`.

### Métodos principais

#### `gerar_pergunta(exercicio, tipo)`

Recebe:

- `exercicio`: objeto `Exercicio`;
- `tipo`: tipo de pergunta.

Faz:

1. monta o prompt com `QuestionBuilder`;
2. chama `_chamar_api()`.

Retorna:

- `list[Pergunta]`, normalmente com uma pergunta.

#### `_chamar_api(numero, titulo, codigo, tipo, prompt)`

Recebe:

- dados imutáveis do exercício;
- tipo da pergunta;
- prompt completo.

Faz:

- monta uma chave de cache;
- chama `_chamar_api_cached()`.

Retorna:

- `list[Pergunta]`.

#### `_chamar_api_cached(chave, prompt)`

Recebe:

- `chave`: tupla com número, título, código e tipo;
- `prompt`: texto enviado ao LLM.

Faz:

- envia uma requisição `chat.completions.create`;
- define mensagem de sistema como tutor de programação;
- usa o modelo e a temperatura configurados;
- extrai o conteúdo textual da resposta;
- chama `_parsear_resposta()`.

Retorna:

- lista de perguntas ou lista vazia em caso de erro.

O decorador `lru_cache(maxsize=200)` evita chamadas repetidas para o mesmo exercício e tipo durante a execução.

#### `_parsear_resposta(texto, exercicio_numero)`

Recebe:

- resposta textual do LLM;
- número do exercício.

Faz:

- tenta extrair JSON;
- percorre os itens retornados;
- cria objetos `Pergunta`.

Retorna:

- `list[Pergunta]`.

#### `_extrair_json(texto)`

Recebe:

- texto bruto retornado pelo LLM.

Faz:

- localiza o primeiro `[` e o último `]`;
- tenta interpretar o trecho como JSON.

Retorna:

- lista de dicionários ou lista vazia.

### Conexões

`AIClient` conecta a lógica local ao modelo de linguagem. Ele depende de `QuestionBuilder`, `Pergunta`, `Exercicio` e das configurações do LM Studio.

## `services/question_orchestrator.py`

### Função do arquivo

Coordena o processo de geração. Ele decide quais exercícios serão usados, quais tipos serão sorteados e quando o LLM será chamado.

### Classe principal

#### `QuestionOrchestrator`

Na criação:

- recebe opcionalmente um `AIClient`;
- se nenhum for fornecido, cria um novo;
- cria também um `Sorteador`.

### Métodos

#### `gerar(exercicios, total)`

Recebe:

- `exercicios`: lista de objetos `Exercicio`;
- `total`: quantidade de perguntas desejadas.

Faz:

1. valida se a lista de exercícios não está vazia;
2. monta os slots de exercícios;
3. sorteia os tipos de pergunta;
4. combina cada exercício com um tipo;
5. chama `AIClient.gerar_pergunta()`;
6. acumula pares `(Exercicio, Pergunta)`.

Retorna:

- `list[tuple[Exercicio, Pergunta]]`.

#### `_montar_slots(exercicios, total)`

Recebe:

- lista de exercícios;
- quantidade total desejada.

Faz:

- se houver exercícios suficientes, usa os primeiros;
- se houver menos exercícios que perguntas, repete os exercícios em ciclo.

Retorna:

- lista de `Exercicio` com tamanho igual a `total`.

### Conexões

Este arquivo é o centro operacional da pasta. Ele conecta `Sorteador`, `AIClient`, `Exercicio` e `Pergunta`.

## `services/report_exporter.py`

### Função do arquivo

Formata e salva o resultado final da geração.

### Classe principal

#### `ReportExporter`

Recebe na criação:

- `caminho_saida`, com valor padrão vindo de `ARQUIVO_SAIDA`.

### Métodos

#### `exportar(pares)`

Recebe:

- lista de tuplas `(Exercicio, Pergunta)`.

Faz:

- chama `_formatar()` para montar o texto;
- chama `_salvar()` para gravar o arquivo.

Retorna:

- não retorna valor.

#### `_formatar(pares)`

Recebe:

- lista de pares `(Exercicio, Pergunta)`.

Faz:

- cria blocos com separador;
- informa o exercício de origem;
- mostra o tipo e o texto da pergunta;
- inclui o código do estudante.

Retorna:

- string com o relatório completo.

#### `_salvar(conteudo)`

Recebe:

- string final do relatório.

Faz:

- cria o diretório de saída, se necessário;
- escreve o arquivo em UTF-8.

Retorna:

- não retorna valor.

### Conexões

É chamado por `main.py` depois da geração. Sua saída é usada como insumo para a etapa de correção.

## Arquivos `__init__.py`

Os arquivos `__init__.py` em `config/`, `models/`, `services/` e `utils/` indicam que essas pastas podem ser tratadas como pacotes Python. Eles não possuem lógica própria relevante, mas ajudam na organização dos imports.

## Pontos mais importantes para entender a lógica

1. `KnowledgeLoader` transforma texto bruto em objetos `Exercicio`.
2. `Sorteador` garante variedade nos tipos de pergunta.
3. `QuestionBuilder` define a intenção pedagógica de cada prompt.
4. `AIClient` envia o prompt ao LLM e interpreta o JSON retornado.
5. `QuestionOrchestrator` une exercícios, tipos e chamadas ao modelo.
6. `ReportExporter` salva o resultado para uso posterior.

Essas partes mostram como o projeto aplica LLMs de forma estruturada: o modelo não é chamado diretamente de qualquer lugar, mas dentro de um fluxo controlado, com entrada organizada, prompt específico e saída padronizada.
