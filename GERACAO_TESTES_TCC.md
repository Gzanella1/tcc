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