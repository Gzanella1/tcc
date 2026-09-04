# AUDITORIA COMPLETA DO USO DE LLM NO PROJETO — TCC -  27/08

**Título da auditoria:** Uso de Grandes Modelos de Linguagem (LLMs) no sistema de tutoria e correção automática de programação.
**Natureza:** Documentação técnica e acadêmica para o Trabalho de Conclusão de Curso.
**Base:** Exclusivamente o código-fonte, arquivos de configuração, documentação e arquivos de resultado disponibilizados no repositório.
**Data da auditoria:** 27/08/2026



# 1. RESUMO EXECUTIVO

O repositório é composto por **três módulos e componentes independentes** que fazem uso de LLM, além de um módulo legado não utilizado:

1. **`gerarPergunta`** — gera perguntas pedagógicas a partir dos exercícios e do código anterior do aluno. Usa **1 chamada à LLM por pergunta** (5 tipos possíveis).
2. **`correcao`** — corrige automaticamente as respostas dos alunos. Usa LLM em **três pontos ativos**:
   - avaliação de respostas textuais (`texto_llm.py`);
   - análise de requisitos em questões de modificação (`modificacao.py`);
   - geração de casos de teste (`tests/generator.py`).
3. **`outros/tutor.py`** — um tutor socrático interativo (chat) que usa LLM.
4. **`correcao/evaluation/correctors.py`** — **código legado/morto**: contém 2 chamadas à LLM, mas **não é importado por nenhum outro módulo**. `[INFERÊNCIA]` trata-se de versão anterior arquivada (o README da pasta a descreve como "referência/versão anterior").

**Modelo principal:** `qwen/qwen3-vl-4b`, servido localmente via LM Studio em `http://localhost:1234/v1`, com API compatível com a OpenAI (endpoint `/chat/completions`).

**Partes determinísticas (sem LLM):** execução de código, verificação de sintaxe, comparação de saída por similaridade, classificação de erros, deduplicação, validação de testes, cálculo de pesos, roteamento de estratégias, geração do relatório de correção e a estratégia de **previsão**.

**Meio-termo:** o sistema é híbrido. A LLM atua apenas como componente semântico; regras objetivas, execução e testes garantem o núcleo determinístico, com múltiplos fallbacks quando a LLM falha.


# 3. MODELO DE LLM UTILIZADO

## 3.1. Módulo `correcao`

Configuração centralizada em `correcao/config.py`:

| Atributo | Valor | Origem |
| --- | --- | --- |
| Nome do modelo | `qwen/qwen3-vl-4b` | `correcao/config.py:21` |
| Variável | `LLM_MODEL` | `correcao/config.py:21` |
| Substituível por variável de ambiente | **Sim** — `os.getenv("LLM_MODEL", "qwen/qwen3-vl-4b")` | `correcao/config.py:21` |
| Base URL | `http://localhost:1234/v1` | `correcao/config.py:20` (variável `LLM_BASE_URL`) |
| Endpoint | `{LLM_BASE_URL}/chat/completions` | `correcao/llm/client.py:34` |
| Timeout | `90` s | `correcao/config.py:22` (variável `LLM_TIMEOUT`) |
| Temperatura padrão | `0.2` | `correcao/llm/client.py:23` (default do parâmetro) |
| `max_tokens` padrão | `1200` | `correcao/llm/client.py:24` (default do parâmetro) |
| API compatível com OpenAI | **Sim** (formato `chat/completions`, `choices[0].message.content`) | `correcao/llm/client.py:34-54` |
| Formato de resposta esperado | Texto contendo JSON de objeto `{...}` | `chamar_llm_json` → `extrair_json` |
| `USAR_LLM` (liga/desliga) | `os.getenv("USAR_LLM", "1")` — desliga com `0/false/no` | `correcao/config.py:23` |

**Diferença entre os elementos (correção de terminologia):**
- **Modelo** = `qwen/qwen3-vl-4b` (nome do modelo).
- **Servidor** = LM Studio (ou qualquer servidor compatível), acessível em `http://localhost:1234`.
- **Biblioteca/API** = API de chat completions compatível com OpenAI; no `correcao` é consumida via `urllib.request` (módulo `llm/client.py`), sem usar a biblioteca `openai`.
- **Interface de protocolo** = HTTP POST com JSON, com cabeçalho `Content-Type: application/json`.

## 3.2. Módulo `gerarPergunta`

Configuração em `gerarPergunta/config/settings.py`:

| Atributo | Valor | Arquivo |
| --- | --- | --- |
| Modelo | `qwen/qwen3-vl-4b` | `gerarPergunta/config/settings.py:9` (`MODEL`) |
| Base URL | `http://localhost:1234/v1` | `config/settings.py:7` (`LM_STUDIO_BASE_URL`) |
| API Key | `lm-studio` | `config/settings.py:8` |
| Temperatura | `0.4` | `config/settings.py:13` (`TEMPERATURE`) |
| `max_tokens` | `512` (constante; atenção: **não é enviada na chamada** — ver §19) | `config/settings.py:14` |
| `MAX_TOKENS` na chamada | **Não enviado** — a chamada em `ai_client.py:93` usa apenas `temperature` | `ai_client.py:93-100` |
| API | Biblioteca `openai` (`OpenAI`) com `base_url` e `api_key` | `ai_client.py:28-31` |

**Observação:** no módulo `gerarPergunta` **não existe variável de ambiente para desligar a LLM nem para substituir o modelo** — os valores estão fixados no código (`settings.py`). `[INFERÊNCIA]` Não há parâmetro `USAR_LLM` neste módulo (ao contrário do módulo `correcao`).

## 3.3. Componente `outros/tutor.py`

Configuração via variáveis de ambiente (com defaults):

| Atributo | Valor | Origem |
| --- | --- | --- |
| Modelo | `local-model` | `tutor.py:18` (`LM_STUDIO_MODEL`) |
| Base URL | `http://localhost:1234/v1` | `tutor.py:16` |
| API Key | `lm-studio` | `tutor.py:17` |
| Temperatura | `0.35` | `tutor.py:21` |
| `max_tokens` | `350` | `tutor.py:108` |
| Timeout | `60` s | `tutor.py:122` |
| Endpoint | `{BASE_URL}/chat/completions` | `tutor.py:111` |
| API | `urllib.request` (sem biblioteca `openai`) | `tutor.py:112` |

---

# 4. CONFIGURAÇÃO DA LLM

## 4.1. Variáveis de ambiente do módulo `correcao`

| Variável | Padrão | Efeito |
| --- | --- | --- |
| `LLM_BASE_URL` | `http://localhost:1234/v1` | URL base do servidor |
| `LLM_MODEL` | `qwen/qwen3-vl-4b` | Modelo |
| `LLM_TIMEOUT` | `90` | Timeout por chamada (segundos) |
| `USAR_LLM` | `1` | `0/false/no` desliga a LLM |

## 4.2. Discrepâncias de configuração

- `correcao` envia `max_tokens` e `temperature` por chamada (parâmetros com default no cliente). `gerarPergunta` **declara** `MAX_TOKENS = 512` mas **não o envia** na chamada (`ai_client.py:93`). `[CRÍTICO]`/`[ALTO]` — inconsistência de configuração (ver §19).
- `gerarPergunta` usa a biblioteca `openai`; `correcao` e `tutor.py` usam `urllib.request` direto. Não é um erro, mas demonstra **dois clientes HTTP diferentes** para o mesmo tipo de servidor.
- O módulo `gerarPergunta` **não** expõe `USAR_LLM`; já o `correcao` expõe. Portanto, se a variável `USAR_LLM=0` for definida no ambiente, ela afeta apenas a correção (não a geração de perguntas nem o tutor).

---

# 5. MAPA DE CHAMADAS À LLM

Todos os pontos em que o código interage com a LLM, no estado atual (ativos e legados):

| ID | Arquivo | Função | Finalidade | Modelo | System Prompt | User Prompt | Temperatura | Max Tokens | Formato esperado |
| -- | ------- | ------ | ---------- | ------ | ------------- | ----------- | ----------- | ---------- | ---------------- |
| LLM-01 | `gerarPergunta/services/ai_client.py` | `_chamar_api_cached` → `chat.completions.create` | Geração de pergunta (5 variações: correção, justificativa, descritiva, modificação, previsão) | `qwen/qwen3-vl-4b` | `Você é um tutor de programação.` | prompt de `QuestionBuilder` (varia por tipo) | `0.4` | *não enviado* | Lista JSON de objetos `{tipo, pergunta}` |
| LLM-02 | `correcao/evaluation/strategies/texto_llm.py` | `avaliar` → `chamar_llm_json` | Correção de resposta textual (usado por `descritiva`, `justificativa`, `correcao` textual e fallback) | `qwen/qwen3-vl-4b` | `Você devolve JSON válido e corrige respostas textuais com justiça.` | prompt de correção textual | `0.15` | `900` | Objeto JSON `{nota, status, feedback, acertos, melhorias}` |
| LLM-03 | `correcao/evaluation/strategies/modificacao.py` | `_avaliar_requisitos_llm` → `chamar_llm_json` | Análise de requisitos em questão de modificação (peso 30%) | `qwen/qwen3-vl-4b` | `Você corrige modificações de código e devolve JSON válido.` | prompt de requisitos de modificação | `0.15` | `1200` | Objeto JSON `{nota, status, cumpre_requisitos, requisitos_identificados, requisitos_atendidos, faltantes, feedback}` |
| LLM-04 | `correcao/tests/generator.py` | `_gerar_testes_llm_once` → `chamar_llm_json` (chamado por `gerar_testes_com_llm`, até 3 tentativas) | Geração de casos de teste | `qwen/qwen3-vl-4b` | `Você gera testes válidos e retorna apenas JSON puro.` | prompt de geração de testes | `0.1` | `1200` | Objeto JSON `{testes: [{entrada, valores_entrada, saida, obs}]}` |
| LLM-05 *(morto)* | `correcao/evaluation/correctors.py` | `avaliar_texto_llm` → `chamar_llm_json` (linha 260) | Correção textual (duplicata legada de LLM-02) | `qwen/qwen3-vl-4b` | `Você devolve JSON válido e corrige respostas textuais com justiça.` | prompt textual (legado) | `0.15` | `900` | JSON `{nota, status, feedback, acertos, melhorias}` |
| LLM-06 *(morto)* | `correcao/evaluation/correctors.py` | `avaliar_modificacao_com_llm` → `chamar_llm_json` (linha 540) | Requisitos de modificação (duplicata legada de LLM-03) | `qwen/qwen3-vl-4b` | `Você corrige modificações de código e devolve JSON válido.` | prompt requisitos (legado) | `0.15` | `1200` | JSON `{nota, status, requisitos...}` |
| LLM-07 | `outros/tutor.py` | `chamar_tutor` (via `urllib`) + reutilizado em `revisar_resposta` | Tutor socrático interativo + revisão de formato | `local-model` (default) | prompt de tutor socrático | input do aluno / conteúdo a revisar | `0.35` | `350` | Texto livre (Pergunta/Dica/Motivação) |

**Observações sobre duplicidade de prompts:**
- LLM-01 possui **5 sub-variações** de user prompt (uma por tipo de pergunta), mas todas compartilham um único system prompt e o mesmo rodapé.
- `descritiva.py` e `justificativa.py` **reesportam** a função `avaliar` de `texto_llm.py` (um único prompt/LLM-02 serve aos dois tipos).
- LLM-02 e LLM-05 são funcionalmente idênticos (prompt duplicado); LLM-03 e LLM-06 idem. As versões LLM-05/06 em `correctors.py` estão mortas.

---

# 6. FLUXO 1 — GERAÇÃO DE PERGUNTAS (`gerarPergunta`)



## 6.2. Detalhadamento por ponto

- **Quem inicia:** `gerarPergunta/main.py:main()`.
- **Função responsável pela geração:** `AIClient.gerar_pergunta()` (`ai_client.py:38`).
- **LLM usada:** model `qwen/qwen3-vl-4b` via biblioteca `openai`.
- **Quantity de perguntas:** `TOTAL_PERGUNTAS = 5` (`settings.py:12`).
- **Cache:** `@lru_cache(maxsize=200)` em `_chamar_api_cached` — chave `(numero, titulo, codigo_base, tipo)`; evita chamar a LLM novamente para o mesmo par (exercício, tipo) quando os exercícios se repetem em ciclo.

## 6.3. Detalhadamento do prompt — LLM-01

### LLM-01 — Geração de pergunta

**Arquivo:** `gerarPergunta/services/ai_client.py` (chamada) + `gerarPergunta/services/question_builder.py` (construção).

**Função:** `AIClient._chamar_api_cached()`; prompt em `QuestionBuilder.construir()`.

**Objetivo:** Gerar 1 pergunta pedagógica de um tipo específico, usando o enunciado original e o código anterior do aluno como contexto.

**System Prompt (constante em `ai_client.py:96`):**
```
Você é um tutor de programação.
```

**User Prompt / template — construído por `QuestionBuilder`.**

Cabeçalho e rodapé comuns (`question_builder.py`):

Rodapé (`_RODAPE`, `question_builder.py:23-30`):
```
Regras obrigatórias:
- Não forneça a resposta
- Não copie o código completo
- Seja claro e objetivo

Retorne APENAS um JSON válido, sem texto extra:
[{"tipo": "{tipo}", "pergunta": "<texto da pergunta>"}]
```

Cabeçalho (`_cabecalho`, `question_builder.py:63-68`):
```
Exercício {exercicio.numero}.

Enunciado original:
{exercicio.enunciado_original}

Código anterior do aluno (contexto/apoio; não é gabarito):
{exercicio.codigo_base}
```

**Variações do corpo (uma por tipo):**

*Correção* (`_correcao`, `question_builder.py:78-84`):
```
Você é um tutor de programação.

Gere 1 pergunta do tipo CORREÇÃO que leve o aluno a identificar
um possível erro lógico, sintático ou de borda no código.

<CABEÇALHO>
<RODAPÉ tipo=correcao>
```

*Justificativa* (`_justificativa`, `question_builder.py:87-93`):
```
Você é um tutor de programação.

Gere 1 pergunta do tipo JUSTIFICATIVA que peça ao aluno para
explicar por que escolheu determinada abordagem no código.

<CABEÇALHO>
<RODAPÉ tipo=justificativa>
```

*Descritiva* (`_descritiva`, `question_builder.py:96-102`):
```
Você é um tutor de programação.

Gere 1 pergunta do tipo DESCRITIVA que peça ao aluno para
descrever, com suas próprias palavras, o que o código faz passo a passo.

<CABEÇALHO>
<RODAPÉ tipo=descritiva>
```

*Modificação* (`_modificacao`, `question_builder.py:105-112`):
```
Você é um tutor de programação.

Gere 1 pergunta do tipo MODIFICAÇÃO que proponha uma alteração
concreta no código (ex.: nova funcionalidade, refatoração, otimização)
e peça ao aluno para pensar como faria essa mudança.

<CABEÇALHO>
<RODAPÉ tipo=modificacao>
```

*Previsão* (`_previsao`, `question_builder.py:115-121`):
```
Você é um tutor de programação.

Gere 1 pergunta do tipo PREVISÃO que apresente uma entrada
específica e pergunte qual será a saída esperada do programa,
incluindo casos de borda se possível.

<CABEÇALHO>
<RODAPÉ tipo=previsao>
```

**Variáveis inseridas:**
- `{tipo}` — tipo da pergunta (correcao/justificativa/descritiva/modificacao/previsao).
- `{exercicio.numero}` — número do exercício.
- `{exercicio.enunciado_original}` — texto do enunciado original.
- `{exercicio.codigo_base}` — código anterior do aluno (contexto).

**Parâmetros (chamada em `ai_client.py:93-100`):**
- `model`: `MODEL` = `qwen/qwen3-vl-4b`.
- `temperature`: `TEMPERATURE` = `0.4`.
- `max_tokens`: **não enviado** (ver §19).
- timeout: não configurado explicitamente na chamada (usa o padrão da biblioteca `openai`).

**Resposta esperada `[ESPERADO]`:** lista JSON, e.g.:
```json
[{"tipo": "correcao", "pergunta": "..."}]
```

**Processamento da resposta (`_parsear_resposta`, `ai_client.py:113-128`):**
1. `_extrair_json` localiza o primeiro `[` e o último `]` e faz `json.loads` (`ai_client.py:131-142`). Se falhar, retorna `[]` (nenhuma pergunta).
2. Para cada item: lê `item.get("tipo")` (default `"desconhecido"`) e `item.get("pergunta")`; se a pergunta não for vazia, cria `Pergunta`.
3. Em `gerar_pergunta` cada `Pergunta` é recriada carregando `enunciado_origem` do exercício (rastreabilidade — Fase 3.1). Campos `conceito_avaliado/evidencia/resposta_esperada/rubrica` permanecem `None`.

**Validação/retry/fallback:**
- **Validação:** apenas verifica se a resposta contém um array JSON e se cada item tem `pergunta` não vazia.
- **Retry:** **não existe** retry nesta chamada.
- **Fallback:** se o `try/except` em `_chamar_api_cached` capturar exceção, retorna `[]` (pergunta perdida) e imprime mensagem no console.
- **Resposta inválida:** retorna `[]` (nenhuma pergunta é gerada para aquele slot).

**Campos do resultado utilizados:** `tipo`, `pergunta`, `exercicio_numero`, e em `gerar_pergunta`: `enunciado_origem` (do exercício, não do LLM).

## 6.4. Parâmetros e contagem de chamadas

- Para `TOTAL_PERGUNTAS = 5`, em condições normais, são feitas **5 chamadas LLM-01** (uma por par exercício/tipo sorteados).
- A quantidade real depende de `TOTAL_PERGUNTAS` e do cache: se houver menos exercícios que slots e o mesmo (exercício, tipo) se repetir, o cache `lru_cache` evita nova chamada.

---

# 7. FLUXO 2 — GERAÇÃO DE CASOS DE TESTE (`correcao/tests/generator.py`)

## 7.1. Quando a geração de testes é acionada

A geração de testes via LLM ocorre em **`obter_testes(q)`** (`tests/generator.py:586`), chamada por:
- `evaluation/strategies/modificacao.py:avaliar()` (linha 220);
- `evaluation/strategies/correcao.py:avaliar()` (linha 48);
- `evaluation/strategies/dispatcher.py` fallback de código (linhas 96/107).

**Fluxo de `obter_testes`:**
1. **`obter_testes_explicitos(q)`** (linha 549) — coleta testes declarados no enunciado/campos da questão (`q.entradaTestes`, `q.saidaTestes`/`q.saida_esperada`, `q.testes`), marcando `_origem = ORIGEM_ENUNCIADO`.
2. Determina a **interface** (`_codigo_para_interface`, linha 58):
   - para `modificacao`/`correcao`: usa `q.codigo_aluno_resposta` (a interface do código atual), senão `q.codigo_base`.
   - para outros tipos: usa `q.codigo_base`.
3. **`usa_input = codigo_tem_input(interface)`** — detecta se o código usa `input()` via regex (`utils/text.py:177`).
4. **Somente se `usa_input == True`**: chama `gerar_testes_com_llm(q, quantidade=TESTES_ALVO)` (linha 602). **Se o código não usa `input()`, NÃO gera testes via LLM** (a régua fica só com os testes explícitos do enunciado).
5. Limpa entradas interativas (`limpar_entrada_interativa`), valida (`validar_testes`), deduplica (`deduplicar_testes`), limita a `TESTES_ALVO` (6).

## 7.2. Como o código determina o uso de `input()` e como isso modifica o prompt

- Determinação: `codigo_tem_input(interface)` — regex `\binput\s*\(` em `utils/text.py:177-181`.
- A assinatura dos `input()` é extraída por `extrair_assinatura_inputs(interface)` (`utils/text.py:468`), que usa AST (com fallback textual para código inválido). A assinatura é descrita no prompt em `_descrever_assinatura_inputs` (`generator.py:79`).
- **Se `usa_input == True`:** o prompt inclui a regra "o código usa input(), gere valores realistas para cada input()", exige que se preencha `valores_entrada` identificado por índice/variável/prompt, e NÃO permite definir a ordem final do stdin (a ordem é reconstruída pelo Python — `_montar_entrada_por_assinatura`).
- **Se `usa_input == False`:** regra "o código NÃO usa input(), então NÃO gere entradas" e `entrada` deve ser `""`.

## 7.3. Prompt completo da geração de testes — LLM-04

### LLM-04 — Geração de casos de teste

**Arquivo:** `correcao/tests/generator.py`.

**Função:** `_gerar_testes_llm_once(q, quantidade)` (linha 407), disparada por `gerar_testes_com_llm` (linha 524).

**Objetivo:** gerar até `TESTES_ALVO` casos de teste (entrada → saída) que reflitam o comportamento correto descrito no **enunciado**, usando o código-base apenas como contexto.

**System Prompt (linha 487):**
```
Você gera testes válidos e retorna apenas JSON puro.
```

**User Prompt / template (linhas 438-483):**
```text
Você é um gerador de testes para código Python.

Tipo da questão:
{q.tipo}

IMPORTANTE:
{regra_input}
- A saída deve refletir o comportamento correto do programa conforme o ENUNCIADO
- O ENUNCIADO é a fonte PRIMÁRIA para determinar o comportamento esperado
- Se o enunciado pedir uma funcionalidade nova (modificação), teste o programa FINAL como descrito no enunciado
- Se o enunciado pedir saída adicional, ela deve aparecer na saída esperada
- NÃO use o código-base para definir a saída esperada — ele é apenas contexto para entender o problema original
- Não invente comportamento fora do enunciado
- Respeite rigorosamente o funcionamento real esperado

Enunciado (fonte dos requisitos):
{q.enunciado}

Código-base (código ANTERIOR do aluno — apenas contexto para entender o problema original; NÃO é gabarito e NÃO define a saída esperada):
{q.codigo_base or "(não fornecido)"}

_INTERFACE_ATUAL_ (interface implementada — define quantos inputs e em que ordem; NÃO define a saída esperada):
{interface or "(não disponível)"}

Gere exatamente {quantidade} testes.

Formato obrigatório (JSON puro):
{
  "testes": [
    {
      "entrada": "",
      "valores_entrada": [
        {"indice": 1, "variavel": "nome_da_variavel", "prompt": "prompt literal se houver", "valor": "valor digitado"}
      ],
      "saida": "saida esperada\n",
      "obs": "descrição do caso"
    }
  ]
}

REGRAS:
{regra_entrada}
- saida deve terminar com \n
- não escreva nada fora do JSON
```

**Variáveis inseridas:**
- `{q.tipo}` — tipo da questão.
- `{regra_input}` — string que orienta sobre input(): `"O código usa input(), então gere valores realistas para cada input()."` **ou** `"O código NÃO usa input(), então NÃO gere entradas."`.
- `{regra_entrada}` — orientações sobre a chave `entrada` (detalhada abaixo).
- `{assinatura_descrita}` — assinatura real dos `input()` (injetada dentro de `regra_entrada` quando há input).
- `{q.enunciado}` — enunciado da questão (fonte dos requisitos).
- `{q.codigo_base}` — código anterior do aluno (contexto).
- `{interface}` — código da interface atual (define quantos/ordem dos inputs).
- `{quantidade}` — `TESTES_ALVO` = 6.

**Quando `usa_input == True`, `regra_input` e `regra_entrada` são (linhas 419-433):**
```
- O código usa input(), então gere valores realistas para cada input().

- NÃO defina a ordem final do stdin na chave "entrada"; ela será montada pelo Python.
- Preencha "valores_entrada" com os valores do cenário, identificados por indice/variavel/prompt.
- A ordem dos objetos dentro de "valores_entrada" não importa.
- O Python montará "entrada" seguindo exatamente esta assinatura real dos input():
{assinatura_descrita}
- NÃO inclua textos dos prompts, menus ou mensagens do programa como valores.
- Exemplo de valores_entrada:
  [
    {"indice": 1, "variavel": "opcao", "valor": "1"},
    {"indice": 2, "variavel": "a", "valor": "5"},
    {"indice": 3, "variavel": "b", "valor": "3"}
  ]
```

**Quando `usa_input == False` (linhas 435-436):**
```
- O código NÃO usa input(), então NÃO gere entradas.
- entrada deve ser "" (string vazia)
```

**Parâmetros (linhas 485-492):**
- `temperature`: `0.1`
- `max_tokens`: `1200`
- modelo: `LLM_MODEL` (`qwen/qwen3-vl-4b`)
- system: `"Você gera testes válidos e retorna apenas JSON puro."`

**JSON esperado `[ESPERADO]`:** objeto com chave `testes` (lista):
```json
{"testes": [{"entrada": "...", "valores_entrada": [...], "saida": "...\n", "obs": "..."}]}
```

**Processamento/validação (`_gerar_testes_llm_once`, linhas 494-521):**
1. Se resposta for dict e `testes` for lista, percorre cada item.
2. Monta a entrada via `_montar_entrada_gerada` (usa valores_entrada estruturada pela assinatura, ou entrada bruta limpa).
3. Se não usa input, força `entrada = ""`.
4. Se usa input e existe assinatura, **descarta** o teste cuja quantidade de blocos de entrada não bata com a quantidade de inputs (`blocos_na_entrada(entrada) != len(assinatura_inputs)`).
5. Anexa `_origem = ORIGEM_LLM`.
6. `validar_testes(...)` → descarta: não-dict, `entrada == None` ou `saida == None`, `saida` vazia, e (se requer input) `entrada` vazia.

**Deduplicação:** `deduplicar_testes` remove testes com entradas duplicadas, mantendo a primeira e preservando a origem.

**Número de tentativas:** `gerar_testes_com_llm` (linha 524) faz **até 3 tentativas**, cada uma uma chamada LLM-04; retorna assim que obtém ≥1 teste válido, caso contrário retorna o "melhor" (maior quantidade obtida).

**O que acontece se a geração falha:**
- Se `USAR_LLM` estiver desativado, retorna `[]` (nenhum teste gerado).
- Se as 3 tentativas não produzirem teste válido, retorna `[]` ou o melhor resultado.
- Em `obter_testes`, se não houver testes no final, retorna `[]` → as estratégias de código lidam com ausência de testes (execução sem entrada; e em `modificacao`, `_resultado_sem_testes`).

## 7.4. Diferença entre "testes fornecidos pelo enunciado" e "testes gerados pela LLM"

- **Testes do enunciado** (`obter_testes_explicitos`): vêm de `entradaTestes`, `saidaTestes`/`saida_esperada` ou `testes` da própria questão. Marcados `_origem = "enunciado"`. Representam a régua oficial declarada.
- **Testes gerados pela LLM** (LLM-04): apenas quando o código usa `input()`. Marcados `_origem = "llm"`. Seu comportamento esperado reflete **a interpretação do LLM sobre o enunciado**.
- **Impacto na correção:** em `evaluation/strategies/codigo.py:avaliar()` (linhas 297-300), a procedência é contada e registrada em `testes_por_origem: {"enunciado": X, "llm": Y}` na evidência de execução. **A nota da execução trata ambos os tipos de teste da mesma forma** (mesma execução e comparação), mas o relatório passa a expor quantos casos vieram de cada origem — rastreabilidade (Etapa 4.4).

---

# 8. FLUXO 3 — CORREÇÃO DE RESPOSTAS

O dispatcher `correcao/evaluation/dispatcher.py:corrigir_questao()` decide a estratégia:

```text
tipo = normalizar_tipo(q.tipo) or inferir_tipo(q.enunciado)
   │
   ├─ validação da entrada (validar_questao) → se erro, retorna Resultado de erro
   │
   ├─ SE tipo == "correcao" E resposta_correcao_eh_textual(q) → texto_llm.avaliar   [LLM]
   │
   ├─ Estratégia por tipo:
   │    previsao      → strategies/previsao.py        (determinístico, SEM LLM)
   │    correcao      → strategies/correcao.py        (pode usar LLM + código/testes)
   │    modificacao   → strategies/modificacao.py     (testes 70% + LLM 30%)
   │    justificativa → strategies/justificativa.py   (= texto_llm, LLM)
   │    descritiva    → strategies/descritiva.py      (= texto_llm, LLM)
   │
   └─ Fallback: questão de tipo desconhecido com indícios de código
         → tenta estrategia_codigo.avaliar
         → se nota 0 (ou sem código), usa texto_llm.avaliar            [LLM]
```

## 8.1. Análise por tipo de questão

| Tipo | LLM participa? | Estratégia | Chamadas LLM |
| --- | --- | --- | --- |
| **previsao** | **Não** | `previsao.py` executa código-base e compara saída | 0 |
| **correcao** (textual) | **Sim** | `texto_llm.avaliar` | LLM-02 |
| **correcao** (código, com testes) | Parcial (na régua) | `correcao.py` → `obter_testes` (LLM-04 se input) + `codigo.py` execução | 0 a 3 (LLM-04) |
| **correcao** (sem testes) | **Sim** | `correcao.py` → fallback para `texto_llm.avaliar` | LLM-02 |
| **modificacao** | **Sim** | `modificacao.py` → testes (LLM-04 se input) + `_avaliar_requisitos_llm` (LLM-03) | LLM-03 + 0 a 3 (LLM-04) |
| **justificativa** | **Sim** | `justificativa.py` = `texto_llm.avaliar` | LLM-02 |
| **descritiva** | **Sim** | `descritiva.py` = `texto_llm.avaliar` | LLM-02 |
| tipo desconhecido | Depende | dispatcher fallback | LLM-02 (± LLM-04) |

## 8.2. Estratégia textual (`texto_llm.py`) — LLM-02

**Fluxo:**
1. `normalizar_texto(q.resposta_aluno)`.
2. Resposta vazia → `Resultado(nota=0, status="erro", feedback="Resposta vazia.")`, fonte `ausente`. *(determinístico, sem LLM)*.
3. `_detectar_conceito_ok(enun_low, resp_low)` — **regra objetiva** (determinística): se o enunciado contém palavras de maiúsculas/minúsculas/padronização, verifica se a resposta contém termos correspondentes. Define `nota_minima` (7.0 se ok, senão 0.0), `status_base`, `feedback_base`.
4. Constrói prompt (abaixo) e chama `chamar_llm_json` (LLM-02, temperatura 0.15, max_tokens 900).
5. Se retornar dict, extrai `nota` (float, com `try/except`), `status`, `acertos`, `melhorias`, `feedback`.
6. **Piso heurístico:** se `conceito_ok`, aplica `nota = max(nota, nota_minima)` e converte `status=="erro"` → `status_base`.
7. Monta `detalhes` (Acertos/Melhorias, limitados a 5 itens) e evidência LLM.
8. Retorna `Resultado` com `nota` limitada a `[0,10]`, fonte `llm`.
9. **Fallback (LLM não retornou dict):** evidência heurística; se `conceito_ok` → nota 7.5, senão nota 3.0, fonte `heuristica`.

**Prompt — LLM-02 (`texto_llm.py:77-107`):**
```text
Você é um corretor de respostas textuais de programação.

Enunciado:
{q.enunciado}

Resposta do aluno:
{q.resposta_aluno or "(vazia)"}

Regras de correção:
- Priorize o CONCEITO, não a gramática.
- Se a ideia principal estiver correta, a nota deve ser alta.
- Erros de português ou frase confusa não devem derrubar muito a nota.
- Se a resposta estiver parcialmente correta, dê nota intermediária.
- Se estiver errada conceitualmente, dê nota baixa.
- Não seja rígido demais com a forma.

Avalie estes itens:
1. entendimento do conceito
2. completude
3. clareza

Retorne APENAS JSON válido neste formato:
{
  "nota": 0,
  "status": "ok|parcial|erro",
  "feedback": "texto curto",
  "acertos": ["..."],
  "melhorias": ["..."]
}
```

**System Prompt (LLM-02):** `"Você devolve JSON válido e corrige respostas textuais com justiça."`

**Variáveis:** `{q.enunciado}`, `{q.resposta_aluno}`.

**JSON esperado `[ESPERADO]`:**
```json
{"nota": <float>, "status": "ok|parcial|erro", "feedback": "<texto>", "acertos": ["..."], "melhorias": ["..."]}
```

**Tratamento de campos/erros:**
- `nota` ausente/sem float → `0.0`; inválida → `0.0`.
- `status` fora do conjunto `{ok, parcial, erro}` → `"parcial"`.
- `acertos`/`melhorias` não-lista → convertidos em lista de 1 elemento.
- `feedback` ausente/vazio → `feedback_base`.
- JSON inválido/None → fallback heurístico (7.5 ou 3.0).

**Resposta → nota:** a nota é atribuída **diretamente pelo LLM** (campo `nota`), eventualmente elevada pelo piso heurístico determinístico. **O LLM atribui nota; o código apenas limita [0,10], aplica piso e valida status/JSON.** Decisão final em caso de sucesso → nota da LLM (com piso); em caso de falha → heurística.

## 8.3. Estratégia de código (`codigo.py`) — execução determinística

**Usada por** `correcao.py` (código com testes) e como componente de 70% de `modificacao.py`.
**Fluxo:** código vazio → nota 0; sintaxe inválida → nota 0 (fonte execucao); sem testes → executa sem entrada; com testes → executa cada caso, compara por similaridade (`comparar_textos` + `saida_contem_esperado`), classifica erros (EOF compatível/incompatível, runtime, timeout), calcula `nota = (passou / total_validos) * 10`, status por fração. **Não usa LLM** (exceto que os próprios testes podem ter sido gerados por LLM em LLM-04).

## 8.4. Estratégia de previsão (`previsao.py`) — SEM LLM

Executa `codigo_base` (referência da execução) para cada par (entrada, saída) extraído da resposta do aluno, ou modo legado. Comparação determinística com `comparar_textos`. **100% sem LLM.** Fonte `execucao`.

---

# 9. ANÁLISE DETALHADA DE MODIFICAÇÃO (`modificacao.py`)

## 9.1. Contrato e origem dos dados

- **Código do aluno a corrigir:** `q.codigo_aluno_resposta` (o NOVO código). É o "objeto corrigido".
- **Código anterior:** `q.codigo_base` — apenas contexto/apoio no prompt; nunca é gabarito, nunca é executado para gerar saída esperada, nunca participa da nota como gabarito.
- **Testes:** `obter_testes(q)` (LLM-04 se input, senão enunciado).

## 9.2. Fluxo da avaliação (`avaliar`, linhas 210-344)

1. `codigo_aluno_resposta = normalizar_texto(q.codigo_aluno_resposta)`.
2. `testes = obter_testes(q)` (pode chamar LLM-04 até 3x se usar input).
3. **Parte objetiva**:
   - se há testes → `avaliar_codigo(q, codigo_aluno_resposta, testes)` (70%);
   - se não há testes e código usa input → `_resultado_sem_testes` (nota 0, fonte ausente — para evitar EOFError falso);
   - se não há testes e código não usa input → `avaliar_codigo` sem testes.
4. Se `USAR_LLM` desativado → retorna apenas a parte objetiva.
5. **`obj = _avaliar_requisitos_llm(q, codigo_aluno_resposta)`** → LLM-03.
6. Se `obj` não é dict → retorna apenas a parte objetiva.
7. Extrai `nota_llm` (float, clamp [0,10]), `status_llm` (validação ok/parcial/erro), requisitos listas, feedback.
8. **Combinação de nota:**
   - **com testes:** `nota_final = 0.7 * resultado_testes.nota + 0.3 * nota_llm`;
   - **sem testes:** `nota_final = nota_llm` (100% LLM) — para evitar que ausência de testes derrube a nota injustamente.
9. **Status final** (com testes): `erro` se `status_llm=="erro"` e testes não-ok; `ok` se ambos ok; senão `parcial`. (sem testes): `status_llm`.
10. **Fonte da evidência:** com testes → `execucao` (predominante, 70%), registrando componente LLM com `peso: 0.3`; sem testes → `llm`.

## 9.3. Prompt — LLM-03 (`modificacao.py:_avaliar_requisitos_llm`, linhas 71-140)

**System Prompt:** `"Você corrige modificações de código e devolve JSON válido."`

**User Prompt:**
```text
Você é um corretor rigoroso de questões de MODIFICAÇÃO de código Python.

=========================
CONTEXTO
=========================

Enunciado (fonte dos requisitos):
{q.enunciado}

O enunciado pede saída/retorno explícito?
{ "SIM" if precisa_saida else "NÃO" }

Código anterior do aluno (apenas contexto/apoio; não é gabarito):
{q.codigo_base or "(não há)"}

Código atual do aluno a ser avaliado:
{codigo_aluno_resposta or "(vazio)"}

=========================
TAREFA
=========================

1. Extraia os REQUISITOS explícitos do enunciado, sem criar requisitos.
2. Para cada requisito:
   - verifique se foi atendido no CÓDIGO ATUAL do aluno;
   - justifique com base no código, sem inventar comportamento.
3. Identifique:
   - o que foi atendido corretamente;
   - o que está incompleto;
   - o que está incorreto.

=========================
REGRAS IMPORTANTES
=========================

- O "Código anterior do aluno" é APENAS contexto/apoio: NÃO é gabarito,
  NÃO define o comportamento esperado e NÃO deve ser corrigido.
  A avaliação é sobre o CÓDIGO ATUAL do aluno frente ao enunciado.
- NÃO invente comportamento que não existe no código.
- NÃO avalie estilo, nome de variável ou formatação.
- Foque APENAS no que o enunciado pede.
- Se o enunciado NÃO pede print/return:
  NÃO penalize ausência de saída.

=========================
CRITÉRIO DE NOTA
=========================

- 10 → todos os requisitos atendidos corretamente.
- 7 a 9 → maioria correta, pequenos problemas.
- 4 a 6 → parcialmente correto.
- 0 a 3 → incorreto ou não atende o requisito principal.

=========================
FORMATO DE SAÍDA
=========================

Retorne APENAS JSON válido neste formato:

{
  "nota": 0,
  "status": "ok|parcial|erro",
  "cumpre_requisitos": true,
  "requisitos_identificados": ["..."],
  "requisitos_atendidos": ["..."],
  "faltantes": ["..."],
  "feedback": "explicação curta e objetiva"
}
```

**Variáveis inseridas:** `{q.enunciado}`, `{precisa_saida}` (SIM/NÃO calculado por `exige_saida_no_enunciado`), `{q.codigo_base}`, `{codigo_aluno_resposta}`.

**Parâmetros:** temperatura 0.15, max_tokens 1200, modelo padrão.

**JSON esperado `[ESPERADO]`:**
```json
{"nota": <float>, "status": "ok|parcial|erro", "cumpre_requisitos": <bool>, "requisitos_identificados": [...], "requisitos_atendidos": [...], "faltantes": [...], "feedback": "..."}
```

**Tratamento de erros:**
- `obj` não dict → retorna apenas parte objetiva (nota dos testes).
- `nota` não-float ou ausente → `0.0`; clamp [0,10].
- `status` inválido → `parcial`.
- `requisitos_atendidos`/`faltantes` usados nos `detalhes` e na evidência (limitados a 6 na exibição).
- Se `USAR_LLM` desativado → parte objetiva pura.
- **Sem testes e sem LLM (ou LLM falhou):** a nota vem apenas da parte objetiva (que pode ser 0 por `_resultado_sem_testes`).

## 9.4. Demonstração matemática da ponderação 70/30

**Com testes:**
> `nota_final = 0.7 × nota_execucao + 0.3 × nota_llm`

Exemplo: testes = 8.0, LLM = 6.0 → `0.7×8 + 0.3×6 = 5.6 + 1.8 = 7.4`.

**Sem testes:**
> `nota_final = nota_llm` (peso 100% LLM)

Essa exceção está explícita em `modificacao.py:288-291` e na docstring: "Sem testes: nota_final = 100% LLM → fonte llm" e "Quando o LLM falha... o Resultado retornado é exatamente o da parte objetiva".

**Nota da LLM** = campo `nota` do JSON (requisitos). **Nota dos testes** = `codigo.py::avaliar` (execução).

---

# 10. INVENTÁRIO COMPLETO DOS PROMPTS

> Os prompts abaixo reproduzem o conteúdo **exatamente** como aparece nos arquivos-fonte. As variáveis de interpolação foram mantidas conforme o código.

---

## PROMPT ID: LLM-01 (Geração de pergunta — 5 sub-variações)

**Arquivo:** `gerarPergunta/services/ai_client.py` + `services/question_builder.py`
**Função:** `AIClient._chamar_api_cached` / `QuestionBuilder.construir`
**Finalidade:** gerar uma pergunta pedagógica por (exercício, tipo).

**SYSTEM:**
```
Você é um tutor de programação.
```

**USER (correção):**
```
Você é um tutor de programação.

Gere 1 pergunta do tipo CORREÇÃO que leve o aluno a identificar um possível erro lógico, sintático ou de borda no código.

Exercício {exercicio.numero}.

Enunciado original:
{exercicio.enunciado_original}

Código anterior do aluno (contexto/apoio; não é gabarito):
{exercicio.codigo_base}

Regras obrigatórias:
- Não forneça a resposta
- Não copie o código completo
- Seja claro e objetivo

Retorne APENAS um JSON válido, sem texto extra:
[{"tipo": "correcao", "pergunta": "<texto da pergunta>"}]
```

**USER (justificativa):** mesmo cabeçalho/rodapé, corpo:
```
Gere 1 pergunta do tipo JUSTIFICATIVA que peça ao aluno para explicar por que escolheu determinada abordagem no código.
```
**USER (descritiva):**
```
Gere 1 pergunta do tipo DESCRITIVA que peça ao aluno para descrever, com suas próprias palavras, o que o código faz passo a passo.
```
**USER (modificação):**
```
Gere 1 pergunta do tipo MODIFICAÇÃO que proponha uma alteração concreta no código (ex.: nova funcionalidade, refatoração, otimização) e peça ao aluno para pensar como faria essa mudança.
```
**USER (previsão):**
```
Gere 1 pergunta do tipo PREVISÃO que apresente uma entrada específica e pergunte qual será a saída esperada do programa, incluindo casos de borda se possível.
```

**Parâmetros:** temperatura `0.4`; max_tokens **não enviado**; modelo `qwen/qwen3-vl-4b`.

---

## PROMPT ID: LLM-02 (Correção textual)

**Arquivo:** `correcao/evaluation/strategies/texto_llm.py`
**Função:** `avaliar`
**Finalidade:** avaliar resposta textual (descritiva/justificativa/correção textual/fallback).

**SYSTEM:**
```
Você devolve JSON válido e corrige respostas textuais com justiça.
```

**USER:**
```
Você é um corretor de respostas textuais de programação.

Enunciado:
{q.enunciado}

Resposta do aluno:
{q.resposta_aluno or "(vazia)"}

Regras de correção:
- Priorize o CONCEITO, não a gramática.
- Se a ideia principal estiver correta, a nota deve ser alta.
- Erros de português ou frase confusa não devem derrubar muito a nota.
- Se a resposta estiver parcialmente correta, dê nota intermediária.
- Se estiver errada conceitualmente, dê nota baixa.
- Não seja rígido demais com a forma.

Avalie estes itens:
1. entendimento do conceito
2. completude
3. clareza

Retorne APENAS JSON válido neste formato:
{
  "nota": 0,
  "status": "ok|parcial|erro",
  "feedback": "texto curto",
  "acertos": ["..."],
  "melhorias": ["..."]
}
```

**Parâmetros:** temperatura `0.15`; max_tokens `900`; modelo `qwen/qwen3-vl-4b`.

---

## PROMPT ID: LLM-03 (Requisitos de modificação)

**Arquivo:** `correcao/evaluation/strategies/modificacao.py`
**Função:** `_avaliar_requisitos_llm`
**Finalidade:** verificar aderência do código do aluno aos requisitos do enunciado (peso 30%).

**SYSTEM:**
```
Você corrige modificações de código e devolve JSON válido.
```

**USER:**
```
Você é um corretor rigoroso de questões de MODIFICAÇÃO de código Python.

=========================
CONTEXTO
=========================

Enunciado (fonte dos requisitos):
{q.enunciado}

O enunciado pede saída/retorno explícito?
{ "SIM" if precisa_saida else "NÃO" }

Código anterior do aluno (apenas contexto/apoio; não é gabarito):
{q.codigo_base or "(não há)"}

Código atual do aluno a ser avaliado:
{codigo_aluno_resposta or "(vazio)"}

=========================
TAREFA
=========================

1. Extraia os REQUISITOS explícitos do enunciado, sem criar requisitos.
2. Para cada requisito:
   - verifique se foi atendido no CÓDIGO ATUAL do aluno;
   - justifique com base no código, sem inventar comportamento.
3. Identifique:
   - o que foi atendido corretamente;
   - o que está incompleto;
   - o que está incorreto.

=========================
REGRAS IMPORTANTES
=========================

- O "Código anterior do aluno" é APENAS contexto/apoio: NÃO é gabarito,
  NÃO define o comportamento esperado e NÃO deve ser corrigido.
  A avaliação é sobre o CÓDIGO ATUAL do aluno frente ao enunciado.
- NÃO invente comportamento que não existe no código.
- NÃO avalie estilo, nome de variável ou formatação.
- Foque APENAS no que o enunciado pede.
- Se o enunciado NÃO pede print/return:
  NÃO penalize ausência de saída.

=========================
CRITÉRIO DE NOTA
=========================

- 10 → todos os requisitos atendidos corretamente.
- 7 a 9 → maioria correta, pequenos problemas.
- 4 a 6 → parcialmente correto.
- 0 a 3 → incorreto ou não atende o requisito principal.

=========================
FORMATO DE SAÍDA
=========================

Retorne APENAS JSON válido neste formato:

{
  "nota": 0,
  "status": "ok|parcial|erro",
  "cumpre_requisitos": true,
  "requisitos_identificados": ["..."],
  "requisitos_atendidos": ["..."],
  "faltantes": ["..."],
  "feedback": "explicação curta e objetiva"
}
```

**Parâmetros:** temperatura `0.15`; max_tokens `1200`; modelo `qwen/qwen3-vl-4b`.

---

## PROMPT ID: LLM-04 (Geração de testes)

**Arquivo:** `correcao/tests/generator.py`
**Função:** `_gerar_testes_llm_once`
**Finalidade:** gerar casos de teste a partir do enunciado.

**SYSTEM:**
```
Você gera testes válidos e retorna apenas JSON puro.
```

**USER:**
```
Você é um gerador de testes para código Python.

Tipo da questão:
{q.tipo}

IMPORTANTE:
{regra_input}
- A saída deve refletir o comportamento correto do programa conforme o ENUNCIADO
- O ENUNCIADO é a fonte PRIMÁRIA para determinar o comportamento esperado
- Se o enunciado pedir uma funcionalidade nova (modificação), teste o programa FINAL como descrito no enunciado
- Se o enunciado pedir saída adicional, ela deve aparecer na saída esperada
- NÃO use o código-base para definir a saída esperada — ele é apenas contexto para entender o problema original
- Não invente comportamento fora do enunciado
- Respeite rigorosamente o funcionamento real esperado

Enunciado (fonte dos requisitos):
{q.enunciado}

Código-base (código ANTERIOR do aluno — apenas contexto para entender o problema original; NÃO é gabarito e NÃO define a saída esperada):
{q.codigo_base or "(não fornecido)"}

_INTERFACE_ATUAL_ (interface implementada — define quantos inputs e em que ordem; NÃO define a saída esperada):
{interface or "(não disponível)"}

Gere exatamente {quantidade} testes.

Formato obrigatório (JSON puro):
{
  "testes": [
    {
      "entrada": "",
      "valores_entrada": [
        {"indice": 1, "variavel": "nome_da_variavel", "prompt": "prompt literal se houver", "valor": "valor digitado"}
      ],
      "saida": "saida esperada\n",
      "obs": "descrição do caso"
    }
  ]
}

REGRAS:
{regra_entrada}
- saida deve terminar com \n
- não escreva nada fora do JSON
```

**Variáveis:** `{q.tipo}`, `{regra_input}`, `{q.enunciado}`, `{q.codigo_base}`, `{interface}`, `{quantidade}`, `{regra_entrada}` (com `{assinatura_descrita}` quando há input).

**Parâmetros:** temperatura `0.1`; max_tokens `1200`; modelo `qwen/qwen3-vl-4b`.

---

## PROMPT IDs: LLM-05 e LLM-06 (legados — NÃO executados)

**Arquivo:** `correcao/evaluation/correctors.py`
**Funções:** `avaliar_texto_llm` (LLM-05) e `avaliar_modificacao_com_llm` (LLM-06).

**LLM-05 — SYSTEM:** `"Você devolve JSON válido e corrige respostas textuais com justiça."`
**USER:** idêntico ao de LLM-02 (mesmo conteúdo, difere apenas na indentação da versão legada — reproduzido em `correctors.py:228-258`).
**Parâmetros:** temperatura `0.15`, max_tokens `900`.

**LLM-06 — SYSTEM:** `"Você corrige modificações de código e devolve JSON válido."`
**USER:** versão com indentação de 8 espaços, conteúdo essencialmente igual ao de LLM-03 (reproduzido em `correctors.py:472-538`).
**Parâmetros:** temperatura `0.15`, max_tokens `1200`.

> Observação importante: **nenhum módulo importa `correctors`**. `[INFERÊNCIA]` Estes prompts são duplicatas mortas, não enviados à LLM na execução atual.

---





# 13. INFLUÊNCIA DA LLM NA NOTA

| Estratégia | LLM participa? | Nota da LLM | Nota do código/testes | Peso LLM | Peso testes | Nota final |
| --- | --- | --- | --- | --- | --- | --- |
| previsao | Não | — | sim (execução) | 0% | 100% | nota calculada pelo código |
| correcao (textual) | Sim | sim (campo `nota`) | — | 100% (com piso heurístico) | 0% | nota LLM limitada [0,10] + piso |
| correcao (código + testes) | Parcial (régua) | não (não há nota LLM) | sim (execução) | 0% (na nota) | 100% | nota de execução dos testes |
| correcao (sem testes) | Sim | sim | — | 100% | 0% | nota LLM texto |
| modificacao (com testes) | Sim | sim (requisitos) | sim (execução) | **30%** | **70%** | `0.7×testes + 0.3×LLM` |
| modificacao (sem testes) | Sim | sim | — (0) | 100% | 0% | `nota_llm` |
| modificacao (LLM falhou/off) | Não | — | sim | 0% | 100% | nota da parte objetiva |
| justificativa | Sim | sim | — | 100% | 0% | nota LLM texto |
| descritiva | Sim | sim | — | 100% | 0% | nota LLM texto |
| tipo desconhecido | Depende | — | — | — | — | fallback (código ou texto LLM) |

**Respostas-chave:**
- **A LLM atribui diretamente a nota?** Sim, nos fluxos textuais (campo `nota`) e na parte de requisitos da modificação (campo `nota`).
- **A LLM apenas fornece informações?** Na modificação ela fornece `requisitos_atendidos/faltantes` e `nota`, mas a nota final combina com testes.
- **A nota é calculada pelo código?** Sim — a nota final é sempre calculada/limitada pelo código (`max(0, min(10, ...))`, `round`, ponderação).
- **A nota da LLM possui peso?** Sim, apenas em modificação com testes (30%); nos fluxos textuais é 100% (sem testes).
- **Nota mínima por heurística?** Sim — em `texto_llm`, piso de 7.0 quando `conceito_ok` é detectado por regra objetiva.
- **Fallback?** Sim — LLM falha → heurística (7.5/3.0) no texto; → parte objetiva na modificação.
- **Nota limitada entre 0 e 10?** Sim — todos os caminhos limitam a `[0,10]`.
- **Quem tem a decisão final?** O **código** valida, limita e pondera; a LLM fornece a nota semântica. Em modificação com LLM ativo, a nota final é sempre `clamp(0.7×testes + 0.3×LLM)` (ou `clamp(LLM)` sem testes).

---



# Exercícios Gerados com Base nas Respostas do Aluno

---

## Exercício 1 — PREVISÃO

**Exercício gerado com base na resposta da questão 1**

### Questão

**[PREVISAO]** Qual será a saída do programa para a entrada `"abc"`?
E como seria a saída para a entrada `"aabb"`? Considere casos de borda como strings vazias ou com apenas um caractere.

### Código-base

```python
def verifica_palavra(s):
    conjuntos = set()
    for carater in s:
        conjuntos.add(carater)
    if len(conjuntos) == len(s):
        return True
    return False

def main():
    s = input("digite uma palavra: ")
    print(verifica_palavra(s))

main()
```

### Resposta 1

```text
True
False
```

---

## Exercício 2 — JUSTIFICATIVA

**Exercício gerado com base na resposta da questão 2**

### Questão

**[JUSTIFICATIVA]** Explique por que a abordagem utilizada para calcular a diferença simétrica entre dois conjuntos é eficiente e correta, mesmo que envolva operações como `union`, `intersection` e `difference`.

### Código-base

```python
def diferenca_simetrica(m, n):
    c1 = m.union(n)
    c2 = m.intersection(n)

    return c1.difference(c2)

def main():
    m = {1, 2, 3, 4, 5, 6}
    n = {1, 2, 7, 8, 9, 10}
    print(diferenca_simetrica(m, n))

main()
```

### Resposta 2

> É eficiente pq ela faz o trabalho de forma mais rapida para ver quais itens tem em uma das lista que n tem na outra sem tem que ficar fazendo varios codigos brutos no meio da fução.

---

## Exercício 3 — MODIFICAÇÃO

**Exercício gerado com base na resposta da questão 3**

### Questão

**[MODIFICACAO]** Adicione uma nova funcionalidade à função `buscaReversa`: em vez de buscar chaves por valor, ela deve buscar valores por chave. Ou seja, dada uma chave, devolve o valor associado. Como você faria essa modificação?

### Código-base

```python
def busca_por_chave(d1, chave):
    return d1.get(chave)


def main():
    d1 = {}
    while True:
        chave = input("Digite uma chave: ")
        if chave == "":
            break
        valor = input("Digite um valor: ")
        d1[chave] = valor

    chave = input("Digite a chave para buscar o valor: ")
    valor = busca_por_chave(d1, chave)
    print(valor)


main()
```

### Resposta 3

*Resposta não fornecida.*

---

## Exercício 4 — DESCRITIVA

**Exercício gerado com base na resposta da questão 4**

### Questão

**[DESCRITIVA]** Descreva, com suas próprias palavras, o que o código faz passo a passo, explicando cada parte do programa e como os números são processados até o final.

### Código-base

```python
lista = []

while True:
    n = int(input("digite um numero:"))
    if n == 0:
        break
    else:
        lista.append(n)

novalista = sorted(lista)

for i in novalista:
    print(i)
```

### Resposta 4

> esta em um loop infinito que pede um numero e se ele for zero para o programa se ele for outro numero adiciona na lista. Apos o loop acabar ele sorteia os numeros de forma aleatoria e salva em uma lista depois imprime os numeros aleatorios da nova lista na tela.

---

## Exercício 5 — CORREÇÃO

**Exercício gerado com base na resposta da questão 5**

### Questão

**[CORRECAO]** O código lê números inteiros até que o usuário digite `0`, mas o número `0` é usado como condição de parada. O que acontece se o usuário digitar `0` no primeiro número? O programa continua ou para corretamente?

### Código-base

```python
lista = []

while True:
    n = int(input("digite um numero:"))
    if n == 0:
        break
    else:
        lista.append(n)

lista.sort(reverse=True)

for i in lista:
    print(i)
```

### Resposta 5

> o programa para corretamente
