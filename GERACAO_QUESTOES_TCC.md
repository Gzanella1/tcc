# Geração automática de questões (módulo `gerarPergunta/`)

Este documento descreve o subsistema de **geração de questões** do tutor de
programação com base exclusivamente no código-fonte de `gerarPergunta/`. O
objetivo é explicar, primeiro, o papel de cada parte do sistema e, depois, como
ela está implementada.

> Ponto de atenção: este documento descreve o código como ele está escrito.
> Afirmações de caráter externo ao código (por exemplo, a qualidade pedagógica
> das perguntas geradas pelo modelo de linguagem) não são avaliadas aqui.

---

## 1. Objetivo do módulo

O módulo de geração de questões é a **primeira etapa** de um fluxo maior: um
tutor de programação que cria perguntas a partir do histórico do aluno,
corrige automaticamente a resposta e gera um relatório. Este módulo é o
responsável por converter o *conhecimento* disponível — um arquivo com
exercícios e o código que o aluno produziu anteriormente — em **novas questões
de estudo**.

Em termos práticos, o módulo:

1. lê o arquivo de conhecimento e separa o enunciado original de cada exercício
   do código anterior escrito pelo aluno;
2. decide quais exercícios serão usados e **qual tipo de pergunta** cada um
   receberá;
3. constrói um *prompt* próprio para cada tipo e solicita ao modelo de
   linguagem (LLM) a criação da pergunta;
4. transforma a resposta do modelo no formato interno usado pelo sistema
   (objetos `Pergunta`);
5. exporta o resultado para um arquivo de texto — `conteudo/perguntasGeradas.txt`
   — que será justamente a **entrada do módulo de correção**.

O problema que esse módulo resolve: em vez de o professor escrever perguntas
manualmente, o sistema reaproveita o histórico de exercícios e respostas já
existentes e solicita a um LLM a elaboração de novas questões. Sem esta etapa,
o restante do fluxo (correção e testes) não tem entrada.

---

## 2. Funcionamento geral

O processo de geração pode ser descrito de ponta a ponta em **cinco etapas**,
todas orquestradas por `gerarPergunta/main.py:main`:

> **entrada** (`conhecimento.txt`) → **leitura e estruturação dos exercícios** →
> **definição dos tipos de pergunta** → **construção do prompt** → **LLM** →
> **conversão e validação** → **exportação** (`perguntasGeradas.txt`).

```
conteudo/conhecimento.txt
        │  KnowledgeLoader.carregar()
        ▼
   list[Exercicio]  (numero, enunciado_original, codigo_base)
        │  QuestionOrchestrator.gerar() + Sorteador.tipos_aleatorios()
        ▼
   (exercicio, tipo) para cada posição
        │  QuestionBuilder.construir() → prompt
        ▼
   AIClient.gerar_pergunta()  →  Pergunta (+ dados_entrada)
        │  [PREVISAO] gerar_entrada_testes() → entrada_testes (prompt + validação)
        ▼
   ReportExporter.exportar()
conteudo/perguntasGeradas.txt   (fronteira de entrada da correção)
```

Cada uma dessas etapas é detalhada na próxima seção.

---

## 3. Fluxo passo a passo

### Etapa 1 — Leitura do conhecimento

**O que acontece:**

O sistema primeiro lê o arquivo `conteudo/conhecimento.txt`, que contém duas
seções: as **Questões** (enunciados originais) e as **Respostas** (o código que
o aluno escreveu anteriormente para cada exercício). Cada questão é
identificada por um número e as respostas aparecem em blocos como `resposta 1 -`.

Essa etapa transforma o conteúdo textual em estruturas de dados utilizáveis
(`Exercicio`). A separação entre **enunciado original** e **código anterior do
aluno** (chamado `codigo_base`) é central para o projeto:

- o *enunciado original* é o material usado pelo LLM para criar a pergunta;
- o `codigo_base` serve **apenas como contexto** sobre o que o aluno fez antes —
  jamais é tratado como gabarito.

> Exemplo: o aluno resolveu inicialmente um exercício de soma (`escreva uma
> função que soma dois números`) e escreveu um código. O sistema pode usar esse
> enunciado original e esse código como base para gerar uma pergunta sobre o
> mesmo programa (por exemplo, pedindo uma modificação ou uma previsão de
> saída).

**Como o código implementa:**

A classe `KnowledgeLoader` (`gerarPergunta/services/knowledge_loader.py`) faz o
parsing. O método `carregar()` (`knowledge_loader.py:47`) lê o arquivo, delega
o parsing às expressões regulares da classe e monta a lista ordenada de
`Exercicio` via `_montar_exercicios()` (`knowledge_loader.py:106`):

- `_RE_ENUNCIADO` reconhece os dois formatos de cabeçalho do arquivo
  (`"N. Título"` legado e `"N - Enunciado"`, que é o formato real);
- `_RE_RESPOSTA` delimita o início de um bloco de código do aluno;
- linhas de enunciado só são reconhecidas **fora** de blocos `resposta`, para
  que linhas de código parecidas com enunciado nunca sejam capturadas como
  título da questão;
- se não houver nenhuma resposta, `_montar_exercicios` levanta `ValueError`.

O modelo `Exercicio` (`gerarPergunta/models/exercicio.py:25`) tem os campos
`numero`, `enunciado_original` e `codigo_base`. Quando o aluno não escreveu
código, o campo recebe o placeholder `"[O ALUNO NÃO ESCREVEU CÓDIGO]"`
(`exercicio.py:15`) — literal que também aparece nos arquivos exportados pela
geração.

### Etapa 2 — Definição dos tipos de pergunta

**O que acontece:**

O sistema precisa decidir **quantos exercícios** serão usados e **qual tipo de
pergunta** cada posição da geração vai receber. Existem cinco tipos possíveis e
o sorteio evita que todos os exercícios recebam o mesmo tipo, garantindo
variedade quando o total de perguntas é múltiplo da quantidade de tipos.

Os cinco tipos de pergunta, cada um com uma intenção diferente:

| Tipo | O que a pergunta pede ao aluno |
|---|---|
| `correcao` | Apontar/corrigir um erro no código anterior (sem inventar erro) |
| `justificativa` | Explicar *por que* escolheu determinada abordagem |
| `descritiva` | Descrever, com as próprias palavras, o que o código faz |
| `modificacao` | Modificar o código (nova funcionalidade, refatoração, otimização) |
| `previsao` | Dizer qual saída um programa produz para uma entrada específica |

**Como o código implementa:**

A classe `QuestionOrchestrator`
(`gerarPergunta/services/question_orchestrator.py`) decide os *slots* no método
`gerar(exercicios, total)` (`question_orchestrator.py:35`). Se há exercícios
suficientes, usa os primeiros; caso contrário, repete a lista em ciclo
(`exercicios[i % qtd]`, em `_montar_slots`). Se a lista estiver vazia, levanta
`ValueError`.

O tipo de cada slot vem de `Sorteador.tipos_aleatorios()`
(`gerarPergunta/utils/sorteador.py`), que embaralha a lista de tipos **sem
repetição dentro de cada bloco de 5**. Os tipos e a quantidade de perguntas são
definidos em `gerarPergunta/config/settings.py`:

```python
TIPOS_PERGUNTA = ["correcao", "justificativa", "descritiva", "modificacao", "previsao"]
TOTAL_PERGUNTAS = 5
```

### Etapa 3 — Construção do prompt

**O que acontece:**

Para cada par (exercício, tipo), o sistema monta a mensagem que será enviada ao
LLM. O prompt declara **explicitamente o papel semântico de cada dado** — o
enunciado original é o material da pergunta e o `codigo_base` é contexto/apoio,
"não é gabarito" — e impõe regras obrigatórias de saída (não fornecer a
resposta, não copiar o código completo, retornar apenas JSON válido).

Esta etapa é relevante porque é aqui que o sistema comunica ao modelo **o que
cada campo significa**, evitando que o LLM use o código anterior do aluno como
resposta esperada.

**Como o código implementa:**

A classe `QuestionBuilder` (`gerarPergunta/services/question_builder.py`)
roteia no método `construir(exercicio, tipo)` (`question_builder.py:36`) para o
método específico do tipo. O `_cabecalho(exercicio)` (`question_builder.py:58`)
é compartilhado e declara os papéis:

```
Exercício {numero}.
Enunciado original: {enunciado_original}
Código anterior do aluno (contexto/apoio; não é gabarito): {codigo_base}
```

O `_rodape(tipo)` (`question_builder.py:70`) contém as regras obrigatórias e o
formato de saída JSON `[{"tipo": ..., "pergunta": ...}]`. Cada tipo tem
particularidades:

- `_correcao()` (`question_builder.py:78`) usa um prompt de "análise rigorosa"
  com a regra **"não invente erro"**: se o código estiver correto, o modelo não
  deve criar uma falsa afirmação de erro apenas para gerar a pergunta;
- `_justificativa()` e `_descritiva()` usam cabeçalho + rodapé;
- `_modificacao()` propõe uma alteração concreta;
- `_previsao()` (`question_builder.py:154`) instrui o modelo a preencher o campo
  **obrigatório** `dados_entrada` (dado semântico estruturado, nunca string de
  stdin pronta) e fornece exemplos de formatos aceitos.

### Etapa 4 — Chamada ao modelo e conversão

**O que acontece:**

O prompt é enviado ao servidor local que hospeda o LLM e a resposta (um JSON com
a pergunta) é convertida em objetos `Pergunta`. Para o tipo `previsao` há uma
etapa extra e importante: o modelo informa um campo `dados_entrada` de forma
**semântica estruturada** (um dicionário ou lista descrevendo a entrada), e o
sistema o converte de forma **100% determinística** em uma string de execução
(`entradaTestes`), validando o resultado executando o código-base isoladamente.
Se essa validação falhar, a pergunta é **descartada na geração** — não chega ao
arquivo de saída.

**Como o código implementa:**

A classe `AIClient` (`gerarPergunta/services/ai_client.py`) instancia o cliente
`OpenAI` apontando para o servidor local (`LM_STUDIO_BASE_URL`, por padrão
`http://localhost:1234/v1`). O método `gerar_pergunta(exercicio, tipo)`
(`ai_client.py:39`) monta o prompt, chama a API, carrega na `Pergunta` o texto
original do enunciado (rastreabilidade — `enunciado_origem`) e, para
`previsao`, chama `gerar_entrada_testes(dados_entrada, codigo_base)` (ver §5).

Detalhes de implementação:

- `_chamar_api` / `_chamar_api_cached` (`ai_client.py:95`, `ai_client.py:107`)
  usam cache `lru_cache(maxsize=200)` por chave
  `(numero, titulo, codigo_base, tipo)`; erros de API resultam em lista vazia
  (nenhuma pergunta é gerada naquele slot);
- `_parsear_resposta` (`ai_client.py:133`) converte o JSON em objetos
  `Pergunta`, mantendo o `dados_entrada` bruto;
- `_extrair_json` (`ai_client.py:155`) extrai o primeiro array JSON da resposta.

O modelo `Pergunta` (`gerarPergunta/models/pergunta.py`) guarda `tipo`,
`pergunta`, `exercicio_numero` e `enunciado_origem`. Os campos opcionais de
avaliação (`conceito_avaliado`, `evidencia_codigo`, `resposta_esperada`,
`rubrica`) **permanecem `None`**: conforme o docstring, "o sistema NÃO inventa
evidência nesta fase". Para `previsao`, os campos são `dados_entrada` (cru, não
persiste na correção) e `entrada_testes` (resultado da transformação).

### Etapa 5 — Exportação

**O que acontece:**

O resultado da geração é formatado e salvo em `conteudo/perguntasGeradas.txt`. O
formato do arquivo é pensado para ser lido por humanos **e** pelo módulo de
correção: cada questão aparece com seu tipo, o enunciado da pergunta gerada e o
bloco `Código-base:` (o código anterior do aluno), e a seção `resposta N -`
fica **reservada em branco** para a resposta futura do aluno. No caso de
PREVISAO, uma linha especial `entradaTestes:` transporta a entrada de execução
determinística calculada na etapa anterior.

**Como o código implementa:**

A classe `ReportExporter` (`gerarPergunta/services/report_exporter.py`)
formata e salva o arquivo via `exportar(pares)` (`report_exporter.py:30`). O
formato gerado por `_formatar` é:

```
============================================================
Exercício gerado com base na sua resposta da questão {N}
============================================================

{idx} - [{TIPO}] {pergunta}

Código-base:
----------------------------------------
{código anterior do aluno}
----------------------------------------

[PREVISAO] entradaTestes: "<json string em linha única>"

resposta {idx} -
```

- o bloco `Código-base:` será lido pelo parser da correção como
  `Questao.codigo_base`;
- para PREVISAO, `json.dumps(pergunta.entrada_testes)` serializa a entrada em
  linha única (`report_exporter.py:80-84`), lida depois como
  `Questao.entradaTestes`;
- `_salvar` (`report_exporter.py:92`) cria o diretório do destino se necessário
  e escreve o arquivo.

---

## 4. Principais componentes

| Componente | Arquivo | Responsabilidade |
|---|---|---|
| `KnowledgeLoader` | `services/knowledge_loader.py` | Lê `conhecimento.txt` e monta os `Exercicio` |
| `QuestionOrchestrator` | `services/question_orchestrator.py` | Distribui os exercícios entre os slots da geração |
| `Sorteador` | `utils/sorteador.py` | Sorteia os tipos sem repetição por bloco |
| `QuestionBuilder` | `services/question_builder.py` | Constrói o prompt específico de cada tipo |
| `AIClient` | `services/ai_client.py` | Chama o LLM, faz o parsing e valida PREVISAO |
| `dados_entrada` | `services/dados_entrada.py` | Transforma `dados_entrada` em `entradaTestes` e valida por execução |
| `ReportExporter` | `services/report_exporter.py` | Formata e salva `perguntasGeradas.txt` |
| `Exercicio` / `Pergunta` | `models/exercicio.py` / `models/pergunta.py` | Modelos de dados do fluxo |

Os três componentes mais importantes do módulo são:

- **`KnowledgeLoader`**: a *fronteira de entrada*. Sem ele, o restante do fluxo
  não teria os exercícios estruturados.
- **`AIClient`**: o coração da geração. Ele concentra a comunicação com o LLM,
  a conversão da resposta para `Pergunta` e, por meio do módulo
  `dados_entrada`, a validação das perguntas de previsão.
- **`ReportExporter`**: a *fronteira de saída*, que materializa o artefato
  consumido pela próxima fase do sistema.

---

## 5. Detalhamento técnico

### 5.1 A transformação determinística `dados_entrada → entradaTestes`

O contrato de geração (docstring de `gerarPergunta/services/dados_entrada.py`)
estabelece que o LLM retorna `dados_entrada` como dado **semântico estruturado**
(`dict`/`list`), nunca como string de stdin pronta. Este módulo converte esse
dado para a string de execução de forma **100% determinística**, sem depender
do LLM nem de reinterpretação do enunciado.

**Serializadores** (percorridos em ordem de precedência, o primeiro que
"aceita" o formato vence):

| Serializador | Condição de aceite | Serialização |
|---|---|---|
| `_SerializadorLinear` | `dados` é `list` | Uma linha por elemento (aninhamentos serializados recursivamente) |
| `_SerializadorMapaComBusca` | `dict` com exatamente um dict aninhado | Pares `chave\nvalor` + **linha em branco** (sentinela que encerra laços `while True`) + escalares restantes |
| `_SerializadorDictEscalares` | `dict` | Uma linha por valor |

O registro `_REGISTRO` (`dados_entrada.py:119`) mantém a precedência
(`registrar_serializador` insere no topo). Se nenhum formato for reconhecido,
`serializar_entrada` levanta `FormatoEntradaDesconhecido`.

**Validação por execução de referência:** `executar_referencia(codigo, entrada,
timeout=5)` (`dados_entrada.py:149`) executa o código-base em **subprocesso
isolado** (`python -I`) em diretório temporário — propositalmente desacoplado
do runner de execução da correção. O fluxo completo em
`gerar_entrada_testes(dados, codigo_base)` (`dados_entrada.py:184`) considera a
entrada válida (`ok=True`) somente se:

1. a entrada serializa (não há `FormatoEntradaDesconhecido`);
2. `entradaTestes` não é vazia;
3. a execução de referência não dá timeout;
4. `returncode == 0`;
5. a saída não é vazia.

Qualquer rejeição → `ok=False` com o motivo, e a pergunta é descartada (mensagem
impressa para `stderr` em `ai_client.py:80-84`).

### 5.2 Configuração

`gerarPergunta/config/settings.py` concentra a configuração:

| Constante | Valor padrão |
|---|---|
| `LM_STUDIO_BASE_URL` | `http://localhost:1234/v1` |
| `LM_STUDIO_API_KEY` | `lm-studio` |
| `MODEL` | `qwen/qwen3-vl-4b` |
| `TOTAL_PERGUNTAS` | `5` |
| `TEMPERATURE` | `0.4` |
| `MAX_TOKENS` | `512` |
| `TIPOS_PERGUNTA` | `["correcao","justificativa","descritiva","modificacao","previsao"]` |
| `ARQUIVO_CONHECIMENTO` | `../conteudo/conhecimento.txt` |
| `ARQUIVO_SAIDA` | `../conteudo/perguntasGeradas.txt` |

---

## 6. Integração com os outros módulos

Este módulo é a **fonte de todo o fluxo**: tudo o que a correção e os testes
farão depois parte do que é gerado aqui. O artefato de fronteira é o arquivo
`conteudo/perguntasGeradas.txt`.

**O que atravessa a fronteira geração → correção:**

1. **Enunciado gerado pela IA** → vira `Questao.enunciado` na correção, e o
   enunciado original da geração é preservado como rastreabilidade em
   `Pergunta.enunciado_origem`, que a correção armazena em
   `Questao.extras["enunciado_origem"]`.
2. **`codigo_base`** = o código anterior do aluno. Na geração ele é declarado
   como "contexto/apoio; não é gabarito"; na correção chega como
   `Questao.codigo_base` e mantém o mesmo papel. Ele **nunca** é o gabarito e
   nunca é usado para gerar a saída esperada dos testes.
3. **`entradaTestes`** = a entrada de execução das perguntas de PREVISAO,
   obrigatória e produzida deterministicamente aqui (validada por execução de
   referência). O campo intermediário `dados_entrada` **não persiste** no
   objeto `Questao` da correção — veja `CORRECAO_TCC.md`.
4. **Anti-autocircularidade**: o código que o aluno produzirá como resposta
   ainda não existe nesta fase; e, na correção, `codigo_aluno_resposta` nunca
   participa da geração da própria régua de avaliação (detalhado em
   `GERACAO_TESTES_TCC.md`).

**O que acontece depois:** o módulo de **correção** lê
`perguntasGeradas.txt` (com as respostas do aluno já preenchidas), valida cada
questão, identifica o tipo e direciona para a estratégia de avaliação adequada.
A **geração de testes** (parte da correção) reconstrói a régua de casos de
teste a partir dos campos `entradaTestes`, `saidaTestes`, `saida_esperada` e
`testes`, complementando com casos gerados por LLM quando a interface do
programa usa `input()`.

---

## 7. Arquivos fora do fluxo atual

O diretório `conteudo/` contém também o arquivo `perguntasGeradas01111.txt`.
Ele não é produzido pelos caminhos de saída atuais da geração
(`ARQUIVO_SAIDA` aponta para `perguntasGeradas.txt`) nem consumido pelo parser
padrão da correção. A referência canônica do projeto é `perguntasGeradas.txt`;
este arquivo não participa do fluxo.