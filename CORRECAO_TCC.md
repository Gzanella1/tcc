# Correção automática de questões (módulo `correcao/`)

Este documento descreve o subsistema de **correção automática** do tutor de
programação com base exclusivamente no código-fonte do diretório `correcao/`.
O objetivo é explicar, primeiro, o papel de cada parte do fluxo e, depois, como
as funcionalidades estão implementadas.

> Ponto de atenção: este documento descreve o código como ele está escrito.
> Aspectos de desempenho do modelo de linguagem e a pertinência pedagógica das
> notas são externos ao escopo desta análise.

---

## 1. Objetivo do módulo

O módulo de correção automática recebe o arquivo de questões gerado na etapa
anterior — `conteudo/perguntasGeradas.txt`, já contendo as respostas que o
aluno forneceu — e produz um **relatório de correção** com nota, status,
feedback, testes executados e evidências para cada questão. Em resumo: a
pergunta foi gerada, o aluno respondeu, e agora o sistema precisa **avaliar a
qualidade dessa resposta** e explicar de onde veio a nota.

O problema que esse módulo resolve: as respostas dos alunos são heterogêneas
(texto livre, código a ser corrigido, código modificado, previsão de saída) e
cada tipo de questão exige uma forma de avaliação diferente. Por isso o sistema
não tem um único "corretor": existe uma **estratégia de avaliação por tipo de
questão** e um conjunto de regras que decide qual estratégia usar. Além da
nota, o resultado carrega **rastreabilidade** — de onde a avaliação veio
(execução de código, LLM ou heurística) e quais evidências a sustentam —, de
modo que o relatório não seja apenas um número.

---

## 2. Funcionamento geral

O fluxo de correção de ponta a ponta é:

> **entrada** (`perguntasGeradas.txt`) → **parsing** → **validação do contrato**
> → **identificação do tipo** → **escolha da estratégia** → **correção**
> (execução/testes/LLM) → **evidências** → **resultado** →
> **relatório** (`conteudo/correcao.txt`)

```
conteudo/perguntasGeradas.txt
        │  parser.carregar_questoes()   (JSON ou blocos de texto)
        ▼
   list[Questao]
        │  validation.validar_questao()  → None ou Resultado
        ▼
   dispatcher.corrigir_questao()
        │  tipo = normalizar_tipo() or inferir_tipo()
        │  rota por tipo → strategies/*
        ▼
   Resultado (nota, status, feedback, detalhes, testes_executados,
              fonte_evidencia, evidencias)
        ▼  runner.executar_codigo_python()  (container, limites)
              + usando a régua de testes (tests/generator.py)
        ▼
   formatter.gerar_relatorio()
conteudo/correcao.txt
```

**Por que existem estratégias diferentes?** Porque o objeto avaliado muda com o
tipo da questão:

- Questões de **código** (`modificacao`, `correcao` de código) podem ser
  avaliadas objetivamente: o código é executado contra casos de teste e a nota
  reflete quantos casos passaram;
- Questões de **previsão** (`previsao`) podem ser conferidas de forma
  determinística: executa-se o programa anterior do aluno e compara-se a saída
  real com a resposta;
- Questões **textuais** (`justificativa`, `descritiva`, `correcao` explicativa)
  não têm execução envolvida e são avaliadas por um LLM com regras de
  correção específicas;
- Questões de **modificação** combinam os dois mundos: além de o código
  funcionar, é preciso que a alteração pedida tenha sido feita.

Cada uma dessas estratégias é explorada no fluxo passo a passo e em detalhe na
seção 5.

---

## 3. Fluxo passo a passo

### Etapa 1 — Carregamento e parsing das questões

**O que acontece:**

O sistema carrega o arquivo de questões e o transforma em uma lista de objetos
`Questao` (o "contrato" interno de dados). O parser entende dois formatos:
**JSON** e o **formato de blocos de texto** usado pela geração. Ele também faz
uma distinção conceitual essencial entre dois códigos que aparecem na questão:

- o **`codigo_base`** é o código **anterior do aluno** — serve como contexto
  para entender a interface original do programa, **não é gabarito, não é
  corrigido e não gera a saída esperada**;
- o **`codigo_aluno_resposta`** é o **novo código produzido pelo aluno** — é o
  objeto da correção, executável e testável. Ele nunca é usado para gerar a
  própria régua de avaliação (princípio da anti-autocircularidade).

Quando o aluno respondeu em texto puro, o campo `resposta_aluno` guarda a
resposta **bruta**; ela só é tratada como código quando está explicitamente
delimitada na resposta (por cercas de Markdown ou rótulo `Código:`), nunca a
partir de texto corrido.

**Como o código implementa:**

`correcao/parsing/parser.py`:

- `carregar_questoes(path)` (`parser.py:589`): tenta **JSON primeiro**; se
  falhar, cai no parser de blocos de texto.
- JSON: `_normalizar_item_json` (`parser.py:171`) converte aliases de campos
  para o contrato canônico. O campo intermediário da geração `dados_entrada` é
  **descartado** — não persiste em `Questao` nem vaza para `extras`.
- Texto: `split_exercicios` (`parser.py:276`) reconhece os cabeçalhos de bloco
  do arquivo gerado; `parse_block` (`parser.py:347`) é a máquina de estados que
  captura o `codigo_base` (linhas delimitadas por hífens após rótulos como
  `Código:`/`Código-base:`), a resposta do aluno (seção `resposta N -`), o
  campo `entradaTestes` das perguntas de previsão e o `enunciado_origem` (que
  vai para `extras`, por não ter campo canônico).
- `parse_tests_field` (`parser.py:305`) interpreta o campo `testes` (lista JSON
  ou linhas `entrada => saida` / `entrada | saida`).

O modelo `Questao` (`correcao/models/questao.py:42`) define o contrato dos
campos: `enunciado`, `resposta_aluno`, `resposta_referencia`, `rubrica`,
`codigo_base`, `codigo_aluno_resposta`, `saida_esperada`, `entradaTestes`,
`saidaTestes`, além de `idx`, `testes` (lista de dicionários) e `extras`
(metadados sem campo canônico).

### Etapa 2 — Validação do contrato

**O que acontece:**

Antes de corrigir, o sistema verifica se a questão possui dados suficientes
para ser avaliada. Em vez de apenas responder "sim/não", a validação distingue
três situações que significam coisas diferentes para o sistema:

| Situação | Significado |
|---|---|
| `erro_entrada` | A **questão em si** está incompleta/malformada |
| `pendente` | A questão é válida, mas o estudante **ainda não respondeu** |
| `inconclusivo` | Há resposta, mas faltam dados para uma correção objetiva confiável |

Em nenhum desses casos é produzida uma nota aparentemente válida. Note que a
ausência de resposta do aluno **não é erro de entrada** — é um estado esperado
("pendente"), já que o arquivo gerado é exportado com a seção de resposta em
branco.

**Como o código implementa:**

`correcao/validation.py:validar_questao` (`validation.py:112`) retorna `None`
quando a questão está íntegra; caso contrário, um `Resultado` com o status
adequado. As regras são específicas por tipo:

- **PREVISAO** exige `{enunciado, codigo_base, resposta_aluno, entradaTestes}`.
  `entradaTestes` é **obrigatória** — ela é produzida e validada na geração
  (`dados_entrada → entradaTestes`), nunca derivada do enunciado na correção;
- **MODIFICACAO** exige `{enunciado, codigo_aluno_resposta}`; o `codigo_base` é
  **opcional** e sua ausência NÃO configura `erro_entrada` (`validation.py:155`);
- **CORRECAO** de código exige `codigo_aluno_resposta`; quando a resposta é
  textual (`resposta_correcao_eh_textual`), valida-se pela presença de
  rubrica/referência e resposta;
- **JUSTIFICATIVA / DESCRITIVA** exigem `resposta_aluno` e `rubrica` ou
  `resposta_referencia`.

A distinção entre "campo do aluno" e "campo da questão" é feita por
`_CAMPOS_DO_ALUNO = {"resposta_aluno", "codigo_aluno_resposta"}`
(`validation.py:38`): a ausência de campos do aluno indica apenas **pendência**,
enquanto a ausência de qualquer outro campo é **erro de entrada**. Questões de
`modificacao`/`correcao` de código sem testes declarados, sem `saida_esperada`
e sem possibilidade de gerar testes tornam-se `inconclusivo`
(`_tem_testes_objetivos` e `_pode_gerar_testes`, `validation.py:45` e `:54`).

### Etapa 3 — Identificação do tipo e roteamento

**O que acontece:**

Nem sempre o tipo da questão vem declarado firmemente no arquivo. O sistema
primeiro tenta **normalizar** o tipo informado (aceitando variações como
"correcao de codigo", "prever", "justificar") e, se não houver tipo, **infere**
o tipo a partir do enunciado (mantendo marcações como `[MODIFICACAO]` ou
reconhecendo palavras-chave). Com o tipo resolvido, a questão é **validada** e
então **direcionada** para a estratégia adequada. Se a questão tem tipo
desconhecido, um caminho de contingência tenta a avaliação por execução de
código e, como último recurso, a avaliação textual via LLM.

Há também uma decisão específica para questões de **correção**: a pergunta pode
pedir um **código corrigido** ou uma **explicação textual do erro**. O metadado
`resposta_formato` (produzido pelo parser em `extras`) é a primeira fonte dessa
decisão; se ausente, aplica-se uma heurística conservadora sobre o enunciado
(marcadores como "qual é o erro", "como corrigir", "explique o erro").

**Como o código implementa:**

`correcao/evaluation/dispatcher.py:corrigir_questao` (`dispatcher.py:77`):

1. `tipo = normalizar_tipo(q.tipo) or inferir_tipo(q.enunciado)` — `utils/tipo.py`;
2. `validar_questao(q)`; se houver erro, esse `Resultado` é retornado direto;
3. se `tipo == "correcao"` e a resposta é textual → delega a
   `texto_llm.avaliar` (`dispatcher.py:92-93`);
4. mapeia o tipo para a estratégia via `_ESTRATEGIAS` (`dispatcher.py:68`):
   `previsao`, `correcao`, `modificacao`, `justificativa`, `descritiva`;
5. fallback para tipo desconhecido com indícios de código no enunciado
   (cercas ` ``` ` ou `input(`): avalia apenas `codigo_aluno_resposta` (ou o
   código explicitamente delimitado na resposta, nunca texto corrido) —
   `_resposta_contem_codigo` e `_extrair_resposta_codigo`
   (`dispatcher.py:39` e `:56`);
6. fallback final: `texto_llm.avaliar(q)`.

### Etapa 4 — Avaliação: as estratégias de correção

#### 4.1 Estratégia textual via LLM (justificativa, descritiva, correção explicativa)

**O que acontece:** quando não há código a executar, o LLM avalia a resposta
com regras específicas: priorizar o **conceito** sobre questões de gramática ou
estilo; não assumir que as afirmações do enunciado são verdadeiras; verificar
tecnicamente o código citado; não penalizar a contestação tecnicamente correta.
O sistema também aplica um **piso objetivo de conceito**: se o enunciado trata
de um conceito verificável (por exemplo, maiúsculas/minúsculas/padronização) e
a resposta demonstra esse conceito, a nota não fica abaixo de `7.0`. Se o LLM
não retornar um JSON válido, o sistema usa um **fallback heurístico** (notas
`7.5`/`3.0` conforme o conceito esteja presente ou não).

**Como o código implementa:** `correcao/evaluation/strategies/texto_llm.py:avaliar`
(`texto_llm.py:51`), usado por `justificativa.py`, `descritiva.py` e pela
correção textual. O piso é aplicado por `_detectar_conceito_ok`
(`texto_llm.py:37`); o prompt com as regras está em `texto_llm.py:80-130`; o
JSON retornado é registrado na evidência LLM (nota original, listas de
acertos/melhorias e piso aplicado); o fallback heurístico está em
`texto_llm.py:202-232`, com `fonte_evidencia = "heuristica"`.

#### 4.2 Estratégia de código por execução (parte objetiva)

**O que acontece:** é a avaliação "de verdade, por execução". O sistema primeiro
rejeita código vazio e código com erro de sintaxe. Sem casos de teste,
simplesmente executa o programa sem entrada (nota 10 sem erro, 0 com erro). Com
casos de teste, executa o código do aluno para cada caso, remove os prompts de
`input()` da saída obtida e compara com a saída esperada por **similaridade**
(limiar 0.90). Um caso pode ser classificado como **incompatível**: quando
ocorre `EOFError` e o código do aluno tem **mais** `input()` do que a quantidade
de entradas fornecidas — nesse caso o teste não é contado nem reduz a nota.
A nota final é `(passou / total de casos válidos) × 10`; se todos os casos são
incompatíveis, a nota é 0 (sem aprovação artificial).

**Como o código implementa:** `correcao/evaluation/strategies/codigo.py:avaliar`
(`codigo.py:101`), usado pela `correcao.py` (com testes) e pela `modificacao.py`
(componente objetivo). Detalhes: sintaxe via `verificar_sintaxe_python`
(`runner.py:83`); execução por caso via `executar_codigo_python` com timeout de
3 s (`codigo.py:187-256`); limpeza de prompts com
`extrair_prompts_input`/`remover_prompts_saida`; comparação por
`comparar_textos`/`saida_contem_esperado` (`utils/text.py`); classificação de
casos em `codigo.py:205-234`; contagem de inputs por `contar_inputs_codigo`.

#### 4.3 Estratégia de MODIFICAÇÃO (ponderação 70/30)

**O que acontece:** questões de modificação precisam verificar se o código do
aluno realmente **incorporou a alteração solicitada** — não basta o programa
funcionar. Por isso o sistema combina duas avaliações:

- **parte objetiva** (70% da nota): execução do novo código contra a régua de
  casos de teste;
- **parte LLM** (30% da nota): avaliação de **aderência aos requisitos** do
  enunciado, extraindo os requisitos explícitos e verificando se o código atual
  os atende (com critério escalonado: 10 → 7-9 → 4-6 → 0-3) e sem penalizar a
  ausência de saída quando o enunciado não a pede.

Se **não houver testes**, a nota final passa a ser **100% LLM** (a ausência de
testes não derruba a nota injustamente). Se o LLM estiver desativado ou falhar,
a nota final é exatamente a parte objetiva.

**Como o código implementa:** `correcao/evaluation/strategies/modificacao.py:avaliar`
(`modificacao.py:210`). O objeto da correção é **exclusivamente**
`codigo_aluno_resposta` (`modificacao.py:215`); os testes vêm de
`tests/generator.obter_testes`; sem testes e com `input()` no código, usa-se um
resultado neutro `_resultado_sem_testes` (fonte `ausente`, evita `EOFError`
falso). A ponderação está em `modificacao.py:288-291`; a checagem de requisitos
em `_avaliar_requisitos_llm` (`modificacao.py:67`).

#### 4.4 Estratégia de CORREÇÃO

**O que acontece:** questões de correção são bifurcadas. Se o enunciado pede
**explicação textual** do erro, o sistema executa opcionalmente o `codigo_base`
(sem entrada) para levar uma **evidência real de execução** ao avaliador
textual — desta forma o LLM vê a saída real do programa junto com a resposta do
aluno. Caso contrário, trata-se de correção **de código**: o sistema obtém os
testes e avalia por execução; sem testes, recorre ao LLM.

**Como o código implementa:** `correcao/evaluation/strategies/correcao.py:avaliar`
(`correcao.py:48`). A decisão textual/código usa `_pergunta_eh_textual`
(`correcao.py:36`); a saída real é registrada em
`extras["saida_real_codigo_base"]` (`correcao.py:49-72`, apenas quando não há
`input()` e sem erro/timeout).

#### 4.5 Estratégia de PREVISÃO

**O que acontece:** questões de previsão são avaliadas de forma **100%
determinística — sem LLM e sem derivar a entrada do enunciado**. O `codigo_base`
(o programa anterior do aluno) é a **referência de execução**. O sistema usa
uma estratégia em ordem de prioridade:

1. **por pares**: se a resposta do aluno declara pares `Entrada: X / Saída: Y`,
   o sistema executa o `codigo_base` para cada par declarado pelo próprio aluno
   e exige **quase exatidão** (limiar 0.99), pois previsão é determinística;
2. **modo legado**: se não há pares, executa o `codigo_base` com o campo
   obrigatório `entradaTestes` e compara a saída com a resposta do aluno por
   similaridade (limiar 0.90, nota na faixa 9.0–10.0 próxima; penalizada em
   timeout/erro).

Sem código identificável, a nota é 0 com status `falha`.

**Como o código implementa:** `correcao/evaluation/strategies/previsao.py:avaliar`
(`previsao.py:287`). A extração de pares é `_extrair_pares_resposta`
(`previsao.py:54`, buscas de "Entrada:"/"Saída:"/`output`/`resultado`),
avaliada por `_avaliar_com_pares` (`previsao.py:126`); o modo legado é
`_avaliar_modo_legado` (`previsao.py:213`). A regra-chave (docstring
`previsao.py:9-13,21-29`): a correção de PREVISAO **nunca** chama o LLM e
**nunca** infere a entrada do enunciado — usa somente `q.entradaTestes` ou as
entradas declaradas pelo aluno.

### Etapa 5 — Execução de código em container

**O que acontece:** todo código do aluno é executado dentro de um **container
descartável** com recursos limitados (tempo de parede, CPU, memória, número de
processos/arquivos e rede desativada), nunca no processo principal do sistema.
Isso reduz bastante o risco operacional de rodar código arbitrário, embora —
como o próprio módulo documenta — não torne a execução 100% segura. Os
desfechos possíveis incluem timeout de parede, limite de CPU atingido,
truncamento de saída e erros de runtime com `stderr` preservada.

**Como o código implementa:** `correcao/execution/runner.py`:

- `executar_codigo_python(codigo, entrada, timeout=3)` (`runner.py:296`) monta
  o comando do container em `_montar_comando_container` (`runner.py:119`) e
  executa com `docker`/`podman` `run --rm --network none --read-only --user
  65534:65534` + limites (`--ulimit cpu`, `--pids-limit`, `--memory` etc.);
- bloqueios antes da execução: código vazio, código/entrada acima do limite de
  bytes e runtime indisponível (`_resolver_runtime`, `runner.py:109` — tenta
  `podman`/`docker`);
- `_coletar_saida_limitada` (`runner.py:180`) lê stdout/stderr até o prazo de
  parede, truncando cada stream no limite;
- desfechos: timeout → `timeout=True`/`erro_execucao="Timeout"`; limite de CPU →
  processo morto pelo kernel (`returncode != 0`); truncamento →
  `stdout_truncated`/`stderr_truncated`; erro de runtime → `returncode != 0`
  com `stderr` preservada;
- `verificar_sintaxe_python` (`runner.py:83`) usa `ast.parse` antes da execução.

### Etapa 6 — Registro das evidências e geração do relatório

**O que acontece:** cada `Resultado` não guarda apenas nota/status/feedback.
Ele registra a **origem da avaliação** (`fonte_evidencia`) e uma lista de
**evidências estruturadas** que explicam como a nota foi obtida (ex.: contagens
de testes por procedência, notas originais do LLM, piso aplicado, similaridade
de saída). Essa rastreabilidade aparece no relatório como uma seção
"Evidências:" — sem isso, o relatório seria apenas um número sem justificativa.

**Como o código implementa:**

- `correcao/evaluation/evidencia.py` define as fontes válidas
  (`execucao`, `llm`, `heuristica`, `ausente`; `normalizar_fonte` em
  `evidencia.py:74` garante que o campo sempre transporte um valor válido) e o
  contrato das evidências: cada item é um dicionário serializável
  `{"tipo", "resumo", "dados"[, "peso"]}`. Ausência de evidência NÃO tem
  categoria própria — `evidencias == []` equivale a `fonte == "ausente"`.
- `correcao/models/questao.py:59` define o `dataclass Resultado` (campos `idx`,
  `tipo`, `nota`, `status`, `feedback`, `detalhes`, `testes_executados`,
  `saida_correta`, `fonte_evidencia`, `evidencias`), com `__post_init__`
  normalizando a fonte.
- `correcao/main.py:main` corrige cada questão com `corrigir_questao(q)`; uma
  exceção inesperada vira um `Resultado` com nota 0 e status `erro`
  (`main.py:62-72`).
- `correcao/report/formatter.py:gerar_relatorio` (`formatter.py:121`) monta o
  relatório: cabeçalho "CORREÇÃO AUTOMÁTICA" com data, total, média geral e
  contagem de aprovadas (nota ≥ 7.0), seguido do bloco de cada questão
  (`formatar_resultado`, `formatter.py:53`) com a seção "Evidências:"
  (`_secao_evidencias`, `formatter.py:34`) e "Testes executados:" (entrada,
  saída esperada, saída obtida, motivo, stderr, timeout por caso).

---

## 4. Principais componentes

| Componente | Arquivo | Responsabilidade |
|---|---|---|
| `main.py` | `correcao/main.py` | Ponto de entrada: carrega, corrige e salva o relatório |
| `Questao` / `Resultado` | `correcao/models/questao.py` | Modelos de entrada e de resultado |
| Parser | `correcao/parsing/parser.py` | Lê JSON ou blocos de texto → `Questao` |
| Validação | `correcao/validation.py` | Valida o contrato antes da correção |
| `corrigir_questao` | `correcao/evaluation/dispatcher.py` | Roteador: escolhe a estratégia |
| Estratégias | `correcao/evaluation/strategies/*` | Avaliadores por tipo de questão |
| Evidência | `correcao/evaluation/evidencia.py` | Normalização e contrato das evidências |
| Runner | `correcao/execution/runner.py` | Execução de código em container com limites |
| Cliente LLM | `correcao/llm/client.py` | Comunicação com o servidor LLM |
| Formatter | `correcao/report/formatter.py` | Geração do relatório |
| `obter_testes` | `correcao/tests/generator.py` | Régua de casos de teste (ver `GERACAO_TESTES_TCC.md`) |
| Utilitários | `correcao/utils/text.py`, `correcao/utils/tipo.py` | Normalização de texto, análise de `input()`/AST, tipos |

Os três componentes mais importantes:

- **Dispatcher**: decide *qual* estratégia julga a questão — é ele que garante
  que o tipo certo de avaliação seja aplicado (ou que questiona o fallback).
- **Estratégias**: os avaliadores propriamente ditos (`codigo`, `texto_llm`,
  `modificacao`, `correcao`, `previsao`). Cada um respeita um contrato comum —
  receber `Questao` e devolver `Resultado`.
- **Runner**: a infraestrutura que torna a execução de código arbitrário
  viável dentro de um sistema de correção, com limites e isolamento.

---

## 5. Detalhamento técnico

### 5.1 Rastreabilidade da evidência por caminho

| Caminho | `fonte_evidencia` | Evidências |
|---|---|---|
| Erro de entrada / pendente / inconclusivo / código vazio / falha interna | `ausente` | `[]` |
| Código: sintaxe, execução sem testes, execução com testes | `execucao` | `{tipo: execucao, resumo, dados: {modo, contagens, testes_por_origem, referencia}}` |
| MODIFICACAO com testes | `execucao` (predominante, 70%) | execucao (peso 0.7) + llm (peso 0.3, com requisitos completos) |
| MODIFICACAO sem testes (LLM) | `llm` | `{tipo: llm, resumo, dados: {nota, status, requisitos_*}}` |
| MODIFICACAO placeholder / LLM indisponível | `ausente` ou a da parte objetiva | conforme a parte objetiva |
| Texto (justificativa/descritiva/correção textual) | `llm` (ou `heuristica` no fallback) | `{tipo: llm\|heuristica, resumo, dados: {nota_original, acertos, melhorias, piso_aplicado}}` |
| PREVISAO (pares ou legado) | `execucao` | `{tipo: execucao, resumo, dados: {modo, similaridade, referencia}}` |

### 5.2 Configuração da execução

`correcao/config.py` concentra a configuração (via variáveis de ambiente):

| Constante | Valor padrão |
|---|---|
| `ARQUIVO_ENTRADA` | `../conteudo/perguntasGeradas.txt` |
| `ARQUIVO_SAIDA` | `../conteudo/correcao.txt` |
| `LLM_BASE_URL` | `http://localhost:1234/v1` |
| `LLM_MODEL` | `qwen/qwen3-vl-4b` |
| `LLM_TIMEOUT` | `90` |
| `USAR_LLM` | `"1"` (desligado se `"0"`/`false`/`no`) |
| `LIMIAR_EXATO` / `LIMIAR_APROX` | `0.99` / `0.90` |
| `TESTES_ALVO` | `6` |
| `EXEC_BACKEND` | `container` |
| `CONTAINER_RUNTIME` | `auto` (podman/docker) |
| `CONTAINER_IMAGE` | `python:3.12-alpine` |

**Limites de execução** (`ExecutionLimits`, `runner.py:44`, defaults de
`config.py:33-45`):

| Limite | Valor padrão |
|---|---|
| `timeout_wall_seconds` | `EXEC_TIMEOUT_SECONDS = 3` |
| `cpu_seconds` | `EXEC_CPU_SECONDS = 3` (na prática `max(3, timeout+1)`) |
| `memory_mb` | `EXEC_MEMORY_MB = 128` |
| `max_processes` | `EXEC_MAX_PROCESSES = 32` |
| `max_open_files` | `EXEC_MAX_OPEN_FILES = 64` |
| `max_code_bytes` / `stdin` / `stdout` / `stderr` | `65536` cada |

### 5.3 Cliente LLM

`correcao/llm/client.py`:

- `chamar_llm(messages, temperature=0.2, max_tokens=1200)` (`client.py:21`):
  requisição ao endpoint `{LLM_BASE_URL}/chat/completions` via `urllib.request`
  (formato compatível com OpenAI), com `model=LLM_MODEL` e timeout
  `LLM_TIMEOUT`. Retorna o conteúdo textual ou `None` em falha, ou quando
  `USAR_LLM` está desativado;
- `chamar_llm_json(...)` (`client.py:59`): envolve `chamar_llm` e parseia o
  JSON com `utils/text.py:extrair_json`.

---

## 6. Integração com os outros módulos

**De onde recebe dados:** o arquivo `conteudo/perguntasGeradas.txt`, produzido
pelo módulo de geração (`GERACAO_QUESTOES_TCC.md`). Ao cruzar essa fronteira, o
conteúdo é convertido em objetos `Questao`:

- `Questao.codigo_base` = o código anterior do aluno (contexto, nunca gabarito);
- `Questao.extras["enunciado_origem"]` = rastreabilidade do enunciado original;
- `Questao.entradaTestes` = entrada determinística das perguntas de PREVISAO
  (obrigatória no contrato de PREVISAO).

**Com quem se relaciona durante a avaliação:** a estratégias de código
recorrem ao módulo de **geração de testes** (`correcao/tests/generator.py`, ver
`GERACAO_TESTES_TCC.md`) por meio de `obter_testes(q)`. Essa régua pode ser
composta de casos explícitos do enunciado e de casos gerados por LLM; a
evidência de execução contabiliza a **procedência** de cada caso
(`testes_por_origem: {"enunciado": X, "llm": Y}`) e o relatório a expõe na
seção "Evidências:". A relação é unidirecional: a correção *consome* a régua,
não a produz.

**Para onde envia dados:** o relatório `conteudo/correcao.txt`, que registra, por
questão, nota, status, feedback, detalhes, testes executados e evidências —
incluindo de quantos casos cada origem (enunciado/LLM) participou. Este arquivo
é o produto final visível do sistema para o professor/aluno.

**O que acontece antes e depois, na visão geral do projeto:**

```
geração (conhecimento.txt → perguntasGeradas.txt)
        → correção (perguntasGeradas.txt → correcao.txt)
          usando a régua de testes (enunciado + contexto → testes → validação
          → deduplicação → casos usados na avaliação)
```

---

## 7. Código não utilizado no fluxo atual

1. **`correcao/evaluation/correctors.py`**: módulo de avaliadores
   legados/concentrados. Não é importado por nenhum outro módulo; o
   `correcao/README.md` o descreve como "referência/versão anterior". Não
   participa do fluxo atual.
2. **`correcao/evaluation/strategies/dispatcher.py`**: duplicata de
   `correcao/evaluation/dispatcher.py` (mesmo docstring e mesmo mapeamento).
   O fluxo real importa `from evaluation.dispatcher import corrigir_questao`;
   é candidato a remoção/limpeza.
3. **`extrair_entradas_de_enunciado`** (`utils/text.py:184`) e
   **`_extrair_sequencia_numeros_previsao`** (`parser.py:703`): utilitários que
   extraem a entrada de PREVISAO a partir do enunciado. No contrato atual a
   entrada é obrigatória e produzida na geração (`entradaTestes`); a correção
   nunca deriva a entrada do enunciado, então esses helpers não são o caminho
   usado pela validação de PREVISAO.













  









































  # Geração, validação e deduplicação de casos de teste (`correcao/tests/generator.py`)

Este documento descreve o subsistema de **geração e gestão da régua de testes**
usada na avaliação de código, com base exclusivamente no código-fonte de
`correcao/tests/generator.py` e na sua integração com as estratégias de
avaliação. O objetivo é explicar primeiro por que os casos de teste existem e,
depois, como são obtidos, validados e utilizados.

> Ponto de atenção: a qualidade pedagógica dos casos gerados pelo modelo de
> linguagem e a completude estatística da amostra gerada são externas a este
> documento, que descreve apenas o comportamento implementado.

---

## 1. Objetivo do módulo

Os **casos de teste** são a forma objetiva de verificar o comportamento do
código do aluno: cada caso é um par *(entrada, saída esperada)* e, quando o
sistema executa o código contra um caso, pode afirmar de forma concreta se o
comportamento está ou não de acordo. Sem essa régua, a avaliação de código
ficaria restrita a caminhos textuais ou a respostas pouco conclusivas.

O problema que este módulo resolve é **de onde vêm esses casos** e **como eles
viram uma régua confiável**, em três frentes:

1. **descobrir** os casos que já estão declarados explicitamente na questão
   (campos como `entradaTestes`, `saidaTestes`, `saida_esperada`, `testes`);
2. **completar** a régua com casos gerados pelo LLM — apenas quando o programa
   do aluno usa `input()`, já que sem interação sobre o programa é insuficiente
   pedir casos ao modelo;
3. **higienizar** o resultado: as entradas vindas do LLM costumam conter textos
   de prompt/menu que precisam ser limpos; casos sem saída ou com entrada
   inválida são descartados; entradas duplicadas são removidas; e o conjunto é
   limitado a um alvo máximo de casos.

O módulo é consumido pelas estratégias de avaliação de código (`codigo.py`,
`modificacao.py`, `correcao.py`). A estratégia de PREVISAO **não** usa este
módulo: nela a entrada é produzida deterministicamente na geração da questão.

---

## 2. Funcionamento geral

O fluxo completo é disparado pela função `obter_testes(q)` e pode ser
resumido como:

> **enunciado + código/contexto** → **testes explícitos** (origem `enunciado`)
> e/ou **geração via LLM** (origem `llm`, só quando a interface usa `input()`)
> → **normalização das entradas** → **validação** → **deduplicação** → **casos
> utilizados na correção** (no máximo `TESTES_ALVO`).

```
obter_testes(q)  [tests/generator.py]
   enunciado/questão (entradaTestes/saidaTestes/saida_esperada/testes)
        │  obter_testes_explicitos()  → origem "enunciado"
        ▼
   detecta a interface do programa (_codigo_para_interface)
        │  usa input()?  ── sim → gerar_testes_com_llm()  → origem "llm"
        │
        ▼
   normalização das entradas (assinatura fixa / dinâmica / limpeza interativa)
        ▼
   validar_testes() → deduplicar_testes() → testes[:TESTES_ALVO]
        ▼
   evaluation/strategies/{codigo, modificacao, correcao}.py
        (testes_por_origem: enunciado/llm registrada na evidência)
```

---

## 3. Fluxo passo a passo

### Etapa 1 — Obter os testes explícitos

**O que acontece:**

O sistema primeiro reúne os casos que **já vêm declarados pela própria
questão**, nas formas: `entradaTestes`+`saidaTestes`, `saida_esperada` (um
oráculo de saída declarado), ou a lista `testes` associada à questão. Esses
casos são marcados com a procedência **`enunciado`**, porque nascem do
material que o professor/enunciado entregou. Um detalhe de interface é
respeitado aqui: se a questão declara uma saída esperada mas o programa usa
`input()`, o caso só é criado se houver também uma entrada — senão a execução
não terminaria.

**Como o código implementa:**

`obter_testes_explicitos(q)` (`generator.py:634`):

- lê `q.entradaTestes`/`q.saidaTestes`/`q.saida_esperada`;
- determina a interface (ver Etapa 2);
- cria o caso com `obs = "Caso explícito do enunciado"` e origem `enunciado`
  (`generator.py:644-659`);
- os casos de `q.testes` são copiados com `_com_origem(dict(t), ORIGEM_ENUNCIADO)`
  (`generator.py:664-666`), para não mutar os dicionários originais da `Questao`;
- retorna já deduplicado.

A procedência é carregada pela chave interna `_CHAVE_ORIGEM = "_origem"`
(`generator.py:52`), com `ORIGEM_ENUNCIADO = "enunciado"` e
`ORIGEM_LLM = "llm"` (`generator.py:50-51`).

### Etapa 2 — Determinar a "interface" do programa

**O que acontece:**

Antes de gerar ou limpar casos, o sistema precisa entender a **interface do
programa**: ele espera entrada do usuário via `input()`? Quantos `input()` há, e
em que ordem? O sistema distingue dois tipos de interface:

- **interface fixa**: um número conhecido de `input()`, com variável e prompt
  identificáveis — a entrada pode ser montada na ordem exata;
- **interface dinâmica**: `input()` dentro de um laço (`for`/`while`), em
  quantidade imprevisível até a chegada de fim de arquivo (EOF) — a contagem
  estática não é confiável.

Aqui aparece o princípio mais importante do módulo, a **anti-autocircularidade**:
o código que está sendo corrigido não pode ser usado para construir a própria
régua de avaliação (senão seria fácil "gabaritar" a nota). Por isso:

- a **saída esperada** sempre provém do **enunciado** (fonte primária do
  comportamento esperado), nunca do código do aluno;
- o `codigo_base` (código anterior) é **apenas contexto**; a geração de LLM é
  orientada a não usá-lo para definir a saída esperada;
- o `codigo_aluno_resposta` (código atual) **nunca participa da régua**, mas a
  **interface declarada nele** é usada para decisões de entrada: detectar se o
  programa usa `input()`, quantos inputs e com quais prompts. Para os tipos
  `modificacao`/`correcao`, essa interface é obtida do próprio código atual do
  aluno (é ele que define a interface do programa a corrigir); para os demais
  tipos, usa-se o `codigo_base`.

**Como o código implementa:**

- `_codigo_para_interface(q)` (`generator.py:62`): retorna `codigo_aluno_resposta`
  (com fallback para `codigo_base`) **somente** quando o tipo é
  `modificacao`/`correcao`; para os demais, retorna `codigo_base`. É essa
  função que materializa a regra de anti-autocircularidade no código;
- análise utilitária em `correcao/utils/text.py`: `codigo_tem_input` (`text.py:177`),
  `codigo_tem_input_em_loop` (`text.py:609`, via AST), e
  `extrair_assinatura_inputs` (`text.py:527`), que produz a assinatura
  determinística de cada `input()` (índice 1-based, variável via AST,
  prompt literal e conversor `int`/`float`/`str`/`bool`).
  `_descrever_assinatura_inputs` (`generator.py:83`) formata cada item como
  `{indice}. variavel=...; prompt=...; conversor=...` para o prompt do LLM.

### Etapa 3 — Gerar testes via LLM (quando a interface usa `input()`)

**O que acontece:**

Quando o programa tem interface com `input()`, o sistema pede ao LLM que gere
casos adicionais. O prompt de geração estabelece as **regras de
comportamento**: a saída esperada deve refletir o comportamento correto
conforme o **enunciado** (fonte primária); o enunciado é o material dos
requisitos; o `codigo_base` é apenas contexto e **não define a saída esperada**;
não se deve inventar comportamento fora do enunciado. As regras de *entrada*
variam com o tipo de interface detectado:

| Cenário | Regra de entrada no prompt |
|---|---|
| Interface **dinâmica** (`input()` em loop) | Preencher a `entrada` com o **STDIN completo** (linha por `input()`), simulando iterações realistas do laço; não usar `valores_entrada` |
| Interface **fixa** com `input()` | Preencher `valores_entrada` com `{indice, variavel, valor}` identificados por índice/variável/prompt (o Python monta a entrada pela assinatura real) |
| Sem `input()` | `entrada` deve ser `""` |

O sistema faz **até 3 tentativas** de chamada e guarda o melhor resultado (pelo
tamanho), retornando assim que obtém ao menos um caso válido. Depois, o
resultado passa por um **pós-processamento**: as entradas são montadas
conforme a interface, a contagem de `input()` é conferida (interface fixa) e a
compatibilidade mecânica é checada por execução (interface dinâmica).

**Como o código implementa:**

`gerar_testes_com_llm(q, quantidade=TESTES_ALVO)` (`generator.py:609`): se
`USAR_LLM` está desativado, retorna `[]`; faz até 3 tentativas de
`_gerar_testes_llm_once` (`generator.py:448`), que monta o prompt com tipo da
questão, regras de comportamento (`generator.py:507-513`), enunciado, código-base
e bloco `_INTERFACE_ATUAL_` ("interface implementada — define quantos inputs e em
que ordem; NÃO define a saída esperada"). O formato de saída exigido é JSON puro
`{"testes": [{"entrada", "valores_entrada", "saida", "obs"}]}` com `saida`
terminando em `\n` (`generator.py:526-543`). A chamada usa `chamar_llm_json`
com `temperature=0.1` e `max_tokens=1200` (`generator.py:546-553`).

Pós-processamento (`generator.py:555-603`):

- interface **fixa**: monta a entrada pela assinatura real
  (`_montar_entrada_gerada`) ou, na ausência de valores estruturados, aplica a
  limpeza interativa; o caso é **descartado** se a contagem de blocos da
  entrada não bate com o número de `input()` (`generator.py:579-585`);
- interface **dinâmica**: normaliza a entrada e **descarta** o caso se ela não
  for mecanicamente compatível — `_entrada_atende_interface` executa a
  interface com a entrada e exige que rode até o fim **sem `EOFError` e sem
  timeout**, sem opinar sobre a correção do resultado (`generator.py:587-593`);
- casos aceitos recebem `"_origem": ORIGEM_LLM`.

### Etapa 4 — Normalizar e limpar as entradas

**O que acontece:**

O LLM costuma devolver entradas "sujas" — por exemplo, uma suposta saída/entrada
ruidosa como `"Digite o primeiro número: 5\nDigite o segundo número: 3\nEscolha
a operação:\n1 - Soma\n...\nDigite a opção desejada: 1"` precisa virar o stdin
real `5\n3\n1`. O sistema limpa esses textos para produzir a entrada de execução.
Para interfaces **dinâmicas** a limpeza é mais conservadora: uma linha em
branco pode ser o próprio valor que encerra o laço, então não é removida.

**Como o código implementa:**

- `limpar_entrada_interativa(entrada, codigo)` (`generator.py:710`) remove
  linhas de menu no formato `N - Texto`/`N. Texto`, prefixos de prompt do
  `input()` (mantendo só o valor digitado) e linhas `"texto: valor"` (mantendo
  apenas o valor) — ver exemplo no docstring (`generator.py:714-727`);
- `_normalizar_entrada_dinamica(entrada_bruta)` (`generator.py:785`) preserva
  linhas em branco, normaliza `\r\n`/`\r` → `\n` e remove apenas uma quebra
  final trivial;
- interface fixa: `_montar_entrada_gerada`/`_montar_entrada_por_assinatura`
  (`generator.py:322`/`:286`) montam o stdin **na ordem real dos `input()`**
  detectada pela assinatura, coletando valores estruturados
  (`_coletar_valores_estruturados`, `generator.py:184`: formato preferencial
  `valores_entrada: [{"variavel", "valor", ...}]`, além de fontes alternativas
  como `inputs`, `entradas`, `valores_por_input`), com fallback para a limpeza
  interativa se faltar algum valor.

### Etapa 5 — Validar

**O que acontece:**

O sistema descarta os casos que não sirvam para avaliação: itens que não são
dicionários, casos com `entrada` ou `saida` vazia após normalização e, quando a
interface usa `input()`, casos sem entrada. A procedência (`_origem`) de cada
caso que sobrevive é preservada.

**Como o código implementa:** `validar_testes(testes, requer_input=False)`
(`generator.py:402`); a reconstrução em `{entrada, saida, obs}` preserva
`_origem` quando presente (`generator.py:438-440`).

### Etapa 6 — Deduplicar e limitar

**O que acontece:**

Como casos explícitos e gerados pelo LLM podem se repetir, o sistema remove
entradas duplicadas mantendo a **primeira ocorrência** e, ao final, entrega no
máximo `TESTES_ALVO` casos (6 por configuração). Sem nenhum caso válido, o
resultado é `[]` e as estratégias que dependiam da régua assumem seus caminhos
alternativos (placeholder ou avaliação textual).

**Como o código implementa:** `deduplicar_testes(testes)` (`generator.py:376`)
compara entradas após `normalizar_texto`; `obter_testes(q)` (`generator.py:671`)
aplica `validar_testes` e `deduplicar_testes` e fatia em `testes[:TESTES_ALVO]`
(`generator.py:700-701`).

---

## 4. Principais componentes

| Função | Responsabilidade |
|---|---|
| `obter_testes(q)` | Fluxo completo: explícitos + LLM + normalização + validação + deduplicação + limite |
| `obter_testes_explicitos(q)` | Reúne os casos declarados na questão (origem `enunciado`) |
| `gerar_testes_com_llm(q)` | Gera casos via LLM (até 3 tentativas), com regras por interface |
| `validar_testes(testes)` | Descarta casos sem entrada/saída utilizáveis |
| `deduplicar_testes(testes)` | Remove entradas duplicadas mantendo a primeira ocorrência |
| `_codigo_para_interface(q)` | Decide qual código fornece a interface do programa (anti-autocircularidade) |
| `limpar_entrada_interativa` / `_normalizar_entrada_dinamica` | Higienizam as entradas vindas do LLM |
| `_entrada_atende_interface` | Checa mecanicamente se a entrada roda até o fim (interface dinâmica) |

O componente mais significativo é **`_codigo_para_interface`**, porque é ele
quem garante, no nível estrutural, que o código que está sendo corrigido nunca
construa a própria régua. As demais funções são etapas do mesmo pipeline.

---

## 5. Detalhamento técnico

### 5.1 Suporte utilitário de análise de código

Funções de `correcao/utils/text.py` usadas pelo módulo:

- `codigo_tem_input(codigo)` (`text.py:177`): presença de `input(`.
- `codigo_tem_input_em_loop(codigo)` (`text.py:609`): `input()` dentro de
  `for`/`while` (via AST) — detecta interface dinâmica.
- `extrair_assinatura_inputs(codigo)` (`text.py:527`): assinatura determinística
  dos `input()` na ordem textual (índice 1-based, variável via AST de
  atribuição, prompt literal, conversor). Com erro de sintaxe, usa o fallback
  textual conservador `_extrair_assinatura_inputs_fallback` (`text.py:477`).
- `extrair_prompts_input(codigo)` (`text.py:553`): apenas prompts **literais**
  do `input()`, sem strings de `print`/comentários.
- `classificar_erro_execucao(stderr, returncode)` (`text.py:254`):
  `ERRO_EOF_INCOMPATIVEL` quando há `EOFError`; `ERRO_RUNTIME` para outras
  exceções; `ERRO_NENHUM` sem erro. Como `EOFError` sozinho não prova
  incompatibilidade, o chamador combina com `contar_inputs_codigo()`
  (`text.py:296`).
- `blocos_na_entrada(entrada)` (`text.py:329`): estima a quantidade de blocos
  de stdin (linhas não vazias).

### 5.2 Constantes e limites

| Item | Valor | Local |
|---|---|---|
| Quantidade alvo de testes | `TESTES_ALVO = 6` | `correcao/config.py` |
| Tentativas de geração LLM | 3 | `generator.py:620` |
| Temperatura / max_tokens da chamada | `0.1` / `1200` | `generator.py:546-553` |
| Origem `enunciado` / `llm` | `"enunciado"` / `"llm"` | `generator.py:50-51` |
| Chave interna de origem | `"_origem"` | `generator.py:52` |
| Timeout da validação mecânica (interface dinâmica) | 3 s | `generator.py:359` |

---

## 6. Integração com os outros módulos

A régua produzida aqui é consumida pelas estratégias de avaliação de código:

1. **`codigo.py:avaliar`** (`evaluation/strategies/codigo.py:101`): recebe os
   `testes` e executa cada caso contra `codigo_aluno_resposta`. Contabiliza a
   **procedência** em `testes_por_origem = {"enunciado": X, "llm": Y}`
   (casos sem `_origem` contam como `enunciado`; `codigo.py:293-301`) e registra
   tudo na evidência de execução (`codigo.py:311-324`). É essa contagem que
   aparece no relatório na seção "Evidências:" (ver `CORRECAO_TCC.md`).
2. **`modificacao.py:avaliar`** (`evaluation/strategies/modificacao.py:210`):
   chama `obter_testes`; com testes, a parte objetiva entra com **peso 0.7** na
   ponderação 70/30 e a evidência de execução carrega `peso: 0.7`.
3. **`correcao.py:avaliar`** (`evaluation/strategies/correcao.py:48`): chama
   `obter_testes` e usa `codigo.avaliar` quando há testes; sem testes, recorre
   ao LLM.
4. **PREVISAO não usa este módulo**: a entrada é `q.entradaTestes`, produzida
   deterministicamente na geração
   (`gerarPergunta/services/dados_entrada.py`; ver `GERACAO_QUESTOES_TCC.md`).

**Rastreabilidade da procedência:** a chave `_origem` marca cada caso e é
preservada por validação e deduplicação — a deduplicação mantém a primeira
ocorrência e, portanto, a origem do primeiro caso. A chave **não vaza** para o
relatório público: o formatter renderiza apenas `entrada`, `saida_esperada`,
`saida_obtida`, `obs`, `motivo` e `teste` por caso
(`correcao/report/formatter.py:97-115`).

**Relação com a geração (antes):** os campos `entradaTestes` (obrigatório para
PREVISAO) e `saidaTestes` nascem na geração das questões; este módulo apenas os
consome como fontes explícitas. O campo intermediário `dados_entrada` da
geração não chega aqui.

---

## 7. Pontos de atenção

1. **Interface dinâmica depende do container**: a validação mecânica
   (`_entrada_atende_interface`) usa `executar_codigo_python`, que requer o
   runtime de container (`podman`/`docker`). Quando o runtime não está
   disponível, o fluxo **não gera casos** para interfaces com `input()` em loop.
2. **Anti-autocircularidade e LLM**: o prompt de geração enfatiza que o
   código-base não define a saída esperada; a garantia **estrutural** (nunca
   usar `codigo_aluno_resposta` para montar a régua) é implementada em
   `_codigo_para_interface` — a aderência do modelo ao prompt é externa ao
   código.
3. **Falhas de LLM**: com `USAR_LLM` desativado ou API indisponível,
   `gerar_testes_com_llm` retorna nenhum caso gerado (só os explícitos são
   usados); as estratégias sem régua objetiva tratam isso conforme descrito em
   `CORRECAO_TCC.md` (avaliação textual via LLM ou placeholder neutro).