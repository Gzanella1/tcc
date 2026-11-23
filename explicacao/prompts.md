# AUDITORIA DOS PROMPTS — GERAÇÃO E CORREÇÃO DE QUESTÕES

**Base:** código-fonte do repositório.
**Data:** 27/08/2026
**Escopo:** Somente os prompts utilizados na **geração de questões** e na **correção de respostas**. Excluído intencionalmente o componente de tutoria (`outros/tutor.py`).

> Convenções: **Prompt** = texto presente no código. **Resposta esperada** = formato definido pela implementação. **Resposta real** = resposta registrada em arquivos de execução. Nenhum prompt, parâmetro ou resposta real foi inventado.

---

## 1. Modelo utilizado

Todos os prompts analisados (geração e correção) utilizam **o mesmo modelo de LLM**, porém empregados em tarefas distintas, com instruções, variáveis de entrada e formatos de saída adaptados a cada tipo de questão.

| Item | Valor |
| --- | --- |
| Modelo | `qwen/qwen3-vl-4b` |
| Versão | versão específica não identificada nos arquivos |
| Servidor/provedor | LM Studio local, `http://localhost:1234/v1`, API OpenAI (`/chat/completions`) |
| Configuração | `correcao/config.py:21` (`LLM_MODEL`); `gerarPergunta/config/settings.py:9` (`MODEL`) |

O mesmo modelo é usado na geração de perguntas (`gerarPergunta`) e na correção (`correcao`), com parâmetros (temperature/max_tokens) e prompts diferentes por tarefa.

---

## 2. Prompts de geração de questões

Fonte: `gerarPergunta/services/question_builder.py` (montagem do prompt) e `gerarPergunta/services/ai_client.py` (envio à LLM, função `_chamar_api_cached`).

**System Prompt comum a todos os tipos de geração** (`ai_client.py:116`):
```
Você é um tutor de programação.
```

**Cabecalho e rodapé comuns** (usados por JUSTIFICATIVA, DESCRITIVA e MODIFICAÇÃO; `question_builder.py`). CORREÇÃO e PREVISÃO têm prompts próprios (ver §2.1 e §2.5):

Cabeçalho:
```
Exercício {exercicio.numero}.

Enunciado original:
{exercicio.enunciado_original}

Código anterior do aluno (contexto/apoio; não é gabarito):
{exercicio.codigo_base}
```

Rodapé:
```
Regras obrigatórias:
- Não forneça a resposta
- Não copie o código completo
- Seja claro e objetivo

Retorne APENAS um JSON válido, sem texto extra:
[{"tipo": "{tipo}", "pergunta": "<texto da pergunta>"}]
```

**Configuração comum da geração:** modelo `qwen/qwen3-vl-4b`; temperature `0.4`; `max_tokens` **não enviado na chamada** (declarado `512` em `settings.py` mas não aplicado).

**Resposta esperada (tipos comuns):** array JSON `[{"tipo": "...", "pergunta": "..."}]`. Utilizada para criar o objeto `Pergunta` (tipo + texto da pergunta), gravado em `perguntasGeradas.txt`. Se a resposta não contiver JSON válido com um item de `pergunta` não vazio, nenhuma pergunta é gerada para aquele slot.

**PREVISÃO:** o item também deve incluir `"dados_entrada"` (`<objeto JSON>`). Na geração (`ai_client.py:75-86`), `dados_entrada` é transformado deterministicamente em `entradaTestes` pela função `gerar_entrada_testes`; a pergunta é **rejeitada na geração** se o dado não produzir entrada executável (formato desconhecido, entrada vazia, ou código que falha, estoura timeout ou não produz saída).

Para JUSTIFICATIVA, DESCRITIVA e MODIFICAÇÃO, o que muda entre os tipos é apenas o **corpo do User Prompt** (a primeira regra de instrução):

### 2.1 CORREÇÃO
**Finalidade:** gerar pergunta que leve o aluno a corrigir um erro **real e verificável** no código; o prompt exige uma validação rigorosa para nunca inventar um erro.

**User Prompt:**
```
Você é um tutor de programação especializado em análise rigorosa de código Python.

Sua tarefa é analisar o código produzido pelo aluno e verificar se existe um erro REAL que possa originar uma pergunta do tipo CORREÇÃO.

REGRA PRINCIPAL:
NUNCA invente um erro apenas para gerar uma pergunta.
Se o código estiver correto, não crie uma falsa afirmação de erro.

COMO ANALISAR O CÓDIGO:
1. Verifique a sintaxe real do código.
2. Verifique a lógica e o comportamento do algoritmo.
3. Verifique condições de borda.
4. Verifique se os identificadores usados são consistentes dentro do próprio código.
5. NÃO considere um identificador errado apenas porque poderia ter outro nome.
6. Por exemplo, se uma variável chamada 'carater' é criada e usada consistentemente como 'carater', isso NÃO é erro. O fato de existir a palavra 'caracter' como outra grafia possível não torna 'carater' incorreto.
7. NÃO classifique diferenças de nomenclatura, estilo ou preferência pessoal como erro.
8. Considere formas matemáticas e lógicas equivalentes como corretas.
9. Só considere algo um erro se puder demonstrar objetivamente, a partir do código, que ele produz comportamento incorreto, erro de execução, erro de sintaxe ou falha em algum caso válido.

VALIDAÇÃO OBRIGATÓRIA:
Antes de formular a pergunta, tente refutar a hipótese de erro.
Pergunte a si mesmo: 'Esse trecho realmente está incorreto ou apenas poderia ser escrito de outra maneira?'
Se estiver correto, NÃO invente um problema.

DADOS PARA ANÁLISE:
Enunciado original:
{exercicio.enunciado_original}

Código produzido pelo aluno:
{exercicio.codigo_base}

CRITÉRIO PARA GERAR A PERGUNTA:
A pergunta somente pode ser gerada se existir um erro real, verificável e explicável diretamente pelo código.

Retorne APENAS JSON válido neste formato:
[{"tipo": "correcao", "pergunta": "<pergunta baseada em erro real>"}]
```

**Particularidades:** não usa o cabeçalho/rodapé comuns; os dados são embutidos na seção "DADOS PARA ANÁLISE".

**Variáveis:** `{exercicio.enunciado_original}`, `{exercicio.codigo_base}`.

### 2.2 JUSTIFICATIVA
**Finalidade:** gerar pergunta que peça ao aluno para explicar por que escolheu determinada abordagem no código.

**User Prompt:**
```
Você é um tutor de programação.

Gere 1 pergunta do tipo JUSTIFICATIVA que peça ao aluno para explicar por que escolheu determinada abordagem no código.

<Cabeçalho>
<Rodapé com tipo=justificativa>
```

**Variáveis:** `{tipo}`, `{exercicio.numero}`, `{exercicio.enunciado_original}`, `{exercicio.codigo_base}`.

### 2.3 DESCRITIVA
**Finalidade:** gerar pergunta que peça ao aluno para descrever, com suas próprias palavras, o que o código faz passo a passo.

**User Prompt:**
```
Você é um tutor de programação.

Gere 1 pergunta do tipo DESCRITIVA que peça ao aluno para descrever, com suas próprias palavras, o que o código faz passo a passo.

<Cabeçalho>
<Rodapé com tipo=descritiva>
```

**Variáveis:** idênticas.

### 2.4 MODIFICAÇÃO
**Finalidade:** gerar pergunta que proponha uma alteração concreta no código (nova funcionalidade, refatoração, otimização) e peça ao aluno para pensar como faria essa mudança.

**User Prompt:**
```
Você é um tutor de programação.

Gere 1 pergunta do tipo MODIFICAÇÃO que proponha uma alteração concreta no código (ex.: nova funcionalidade, refatoração, otimização) e peça ao aluno para pensar como faria essa mudança.

<Cabeçalho>
<Rodapé com tipo=modificacao>
```

**Variáveis:** idênticas.

### 2.5 PREVISÃO
**Finalidade:** gerar pergunta que apresente uma entrada específica e pergunte qual será a saída esperada do programa, incluindo casos de borda se possível.

**User Prompt:**
```
Você é um tutor de programação.

Gere 1 pergunta do tipo PREVISÃO que apresente uma entrada específica e pergunte qual será a saída esperada do programa, incluindo casos de borda se possível.

<Cabeçalho>

CAMPO 'dados_entrada' (OBRIGATÓRIO para PREVISAO):
'dados_entrada' deve conter, de forma ESTRUTURADA e semântica, os dados necessários para executar o código-base com a entrada escolhida. NÃO é uma string de stdin — é o dado cru (dict/list). O sistema de correção o converte deterministicamente na entrada de execução.
Exemplos de formato aceito:
  {"dicionario": {"a": 5, "b": 3, "c": 5}, "valor_busca": "5"}
  {"numeros": [2, 3, 4]}
  ["João", "14:00"]
Garanta consistência entre o código-base, o enunciado e 'dados_entrada': o enunciado pode explicar os dados, mas 'dados_entrada' é a fonte estruturada da entrada.

Regras obrigatórias:
- Não forneça a resposta
- Não copie o código completo
- Seja claro e objetivo

Retorne APENAS um JSON válido, sem texto extra, neste formato:
[{"tipo": "previsao", "pergunta": "<texto da pergunta>", "dados_entrada": <objeto JSON com os dados>}]
```

**Particularidades:** usa o cabeçalho comum, porém rodapé próprio com o campo obrigatório `dados_entrada`.

**Variáveis:** `{exercicio.numero}`, `{exercicio.enunciado_original}`, `{exercicio.codigo_base}`.

> **Comparação entre geração de tipos:** JUSTIFICATIVA, DESCRITIVA e MODIFICAÇÃO são estruturalmente idênticos (mesmo system prompt, mesmo cabeçalho, mesmo rodapé e mesmo formato de resposta), diferenciando-se apenas na **instrução central**. **CORREÇÃO** e **PREVISÃO** têm prompts especializados: CORREÇÃO exige análise rigorosa para só gerar pergunta a partir de um erro real e verificável; PREVISÃO mantém o cabeçalho comum, mas obriga o campo estruturado `dados_entrada`. As variáveis fornecidas à LLM são, em todos os casos, o enunciado original e o código-base do aluno.

---

## 3. Prompts utilizados na correção

### 3.1 Correção textual (compartilhado)

**Tipo de questão:** DESCRITIVA, JUSTIFICATIVA, CORREÇÃO textual (quando textual) e fallback de tipo desconhecido.

> **Prompt de correção textual compartilhado.** Utilizado pelos tipos: **DESCRITIVA, JUSTIFICATIVA e CORREÇÃO textual** (e como fallback).
> Os módulos `descritiva.py` e `justificativa.py` reexportam diretamente a função `avaliar` de `texto_llm.py` — por isso o mesmo prompt serve a três tipos, sem duplicação.

**Arquivo:** `correcao/evaluation/strategies/texto_llm.py`
**Função:** `avaliar`
**Modelo:** `qwen/qwen3-vl-4b`

**System Prompt:**
```
Você devolve JSON válido e corrige respostas textuais com justiça.
```

**User Prompt:**
```
Você é um corretor de respostas textuais de programação.

Enunciado:
{q.enunciado}

Código-base:
{q.codigo_base or "(não fornecido)"}

Evidência objetiva de execução do código-base:
{saida_real or "(não disponível)"}

Resposta do aluno:
{q.resposta_aluno or "(vazia)"}

Regras de correção:
- Avalie a resposta com base na correção técnica.
- Priorize o conceito, não a gramática.
- Não assuma que as afirmações presentes no enunciado são necessariamente verdadeiras.
- Analise tecnicamente o código-base antes de avaliar a resposta.
- Se o enunciado afirmar que existe um erro, verifique se esse erro realmente existe.
- Se o aluno contestar a premissa do enunciado, verifique se a contestação é tecnicamente correta.
- Se o aluno demonstrar corretamente que o suposto erro não existe, considere isso um acerto.
- Não penalize o aluno simplesmente por discordar do enunciado.
- A evidência de execução é um dado observado diretamente do código-base.
- Use a evidência de execução como apoio à análise.
- Não conclua que um algoritmo está correto ou incorreto apenas com base em uma única execução.
- Se houver conflito entre a premissa do enunciado e a análise técnica do código, priorize a análise técnica.

Antes de determinar a nota:
1. Identifique o conceito técnico avaliado.
2. Verifique se o suposto erro realmente existe no código-base.
3. Determine se o código implementa corretamente o conceito.
4. Avalie a resposta do aluno.
5. Caso o aluno conteste o enunciado, verifique se a contestação é tecnicamente válida.
6. Só então determine a nota.

Avalie:
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

**Particularidades:** o prompt fornece também o **código-base** e a **evidência objetiva de execução do código-base** (`saida_real`), e orienta a correção pela técnica: o corretor deve verificar se o erro apontado pelo enunciado realmente existe, priorizar a análise técnica sobre a premissa do enunciado e considerar contestação tecnicamente correta como acerto.

**Variáveis do prompt:**

```text
PROMPT

Enunciado:
{q.enunciado}

Código-base:
{q.codigo_base or "(não fornecido)"}

Evidência objetiva de execução do código-base:
{saida_real or "(não disponível)"}

Resposta do aluno:
{q.resposta_aluno or "(vazia)"}
```

| Variável | Origem | Conteúdo |
| --- | --- | --- |
| `q.enunciado` | Questão | Enunciado da questão |
| `q.codigo_base` | Questão | Código anterior do aluno (exibido como `(não fornecido)` se ausente) |
| `saida_real` | Questão (`q.extras["saida_real_codigo_base"]`) | Saída real observada da execução do código-base (exibida como `(não disponível)` se ausente) |
| `q.resposta_aluno` | Questão | Resposta textual enviada pelo aluno (exibida como `(vazia)` se ausente) |

**Parâmetros:** temperature `0.15`; max_tokens `900`.

**Resposta esperada:** objeto JSON `{"nota", "status", "feedback", "acertos", "melhorias"}`.
**Utilização da resposta:** o campo `nota` define a nota da questão (limitada a [0,10]); `status` define o status (valores fora de `ok|parcial|erro` normalizados para `parcial`); `acertos`/`melhorias` alimentam os detalhes; `feedback` o feedback. Há ainda uma regra determinística de piso (se o conceito central do enunciado é detectável na resposta, a nota não fica abaixo de 7.0).

### 3.2 Modificação — avaliação de requisitos

**Tipo de questão:** MODIFICAÇÃO (etapa de correção).

**Arquivo:** `correcao/evaluation/strategies/modificacao.py`
**Função:** `_avaliar_requisitos_llm`
**Modelo:** `qwen/qwen3-vl-4b`

**System Prompt:**
```
Você corrige modificações de código e devolve JSON válido.
```

**User Prompt:**
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

**Variáveis do prompt:**

| Variável | Origem | Conteúdo |
| --- | --- | --- |
| `q.enunciado` | Questão | Enunciado (fonte dos requisitos) |
| `precisa_saida` | derivado do enunciado (`exige_saida_no_enunciado`) | `"SIM"` ou `"NÃO"` indicando se o enunciado pede saída/retorno explícito |
| `q.codigo_base` | Questão | Código anterior do aluno (contexto; não gabarito) |
| `codigo_aluno_resposta` | Questão | Código atual do aluno, objeto da avaliação |

**Parâmetros:** temperature `0.15`; max_tokens `1200`.

**Resposta esperada:** objeto JSON `{"nota", "status", "cumpre_requisitos", "requisitos_identificados", "requisitos_atendidos", "faltantes", "feedback"}`.

**Utilização da resposta:** o campo `nota` compõe a nota final. Na correção de MODIFICAÇÃO, a nota é **combinada**: com testes, `nota_final = 0.7 × nota_dos_testes + 0.3 × nota_llm`; sem testes, `nota_final = nota_llm`. Os campos `requisitos_atendidos`/`faltantes` alimentam os detalhes e a evidência; `feedback` o feedback. `status` fora de `ok|parcial|erro` é normalizado para `parcial`. A execução dos testes em si não depende da LLM.

### 3.3 Outros tipos de questão

- **PREVISÃO — não utiliza LLM na etapa de correção.** A avaliação é realizada por método determinístico (execução do código-base e comparação da saída prevista com a real, via similaridade). Não existe prompt de correção para previsão.
- **CORREÇÃO (quando a resposta é código + há testes):** a correção usa a execução determinística de código contra os testes (sem prompt de correção textual). A LLM entra, nesse fluxo, apenas na **geração de testes** (ver §4). Somente quando a questão de CORREÇÃO é textual ou não há testes é que se usa o prompt de correção textual compartilhado (§3.1).

---

## 4. Geração de casos de teste via LLM

A LLM também é utilizada para **gerar casos de teste** que servem de régua para a correção de questões de código (CORREÇÃO e MODIFICAÇÃO). Só é acionada quando o código da questão usa `input()`.

**Arquivo:** `correcao/tests/generator.py`
**Função:** `_gerar_testes_llm_once` (disparada por `gerar_testes_com_llm`, até 3 tentativas)
**Modelo:** `qwen/qwen3-vl-4b`

**System Prompt:**
```
Você gera testes válidos e retorna apenas JSON puro.
```

**User Prompt (núcleo; trechos condicionais em função do uso de `input()`):**
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

**Variáveis do prompt:**

| Variável | Origem | Conteúdo |
| --- | --- | --- |
| `q.tipo` | Questão | Tipo da questão (ex.: correcao, modificacao) |
| `regra_input` | derivado do código | orientação sobre `input()`, em 3 casos: interface **dinâmica** em laço / "o código usa input(), então gere valores realistas para cada input()" / "o código NÃO usa input(), então NÃO gere entradas" |
| `q.enunciado` | Questão | Enunciado (fonte dos requisitos) |
| `q.codigo_base` | Questão | Código anterior do aluno (contexto) |
| `interface` | derivado do código | interface atual (quantos/ordem dos inputs) |
| `regra_entrada` | derivado | regras da chave `entrada`/`valores_entrada` conforme o caso: laço (stdin completo na `entrada`, sem `valores_entrada`) / assinatura fixa (`valores_entrada`, com a ordem montada pelo Python) / sem input (`entrada` = `""`) |
| `quantidade` | constante | `TESTES_ALVO` = 6 |

**Parâmetros:** temperature `0.1`; max_tokens `1200`.

**Resposta esperada:** objeto JSON com chave `testes` (lista de objetos com `entrada`/`valores_entrada`, `saida`, `obs`).

**Utilização da resposta:** os testes gerados são validados e deduplicados pelo sistema (sem detalhar o algoritmo) e marcados internamente como de origem `llm`, passando a integrar a régua de execução junto com os testes declarados no enunciado (origem `enunciado`). A procedência de cada teste é contabilizada e exposta na evidência de execução (`testes_por_origem`), permitindo distinguir, na correção, quais casos vieram do enunciado e quais foram gerados pela LLM.

---

## 5. Variáveis utilizadas nos prompts (consolidado)

| Variável | Onde é usada | Origem | Conteúdo |
| --- | --- | --- | --- |
| `{tipo}` | Geração (rodapé) | parâmetro da função | Tipo da pergunta (correcao/justificativa/descritiva/modificação/previsão) |
| `{exercicio.numero}` | Geração (cabeçalho) | Exercício | Número do exercício |
| `{exercicio.enunciado_original}` | Geração (cabeçalho) | Exercício | Enunciado original da questão |
| `{exercicio.codigo_base}` | Geração (cabeçalho) | Exercício | Código anterior do aluno (contexto) |
| `{q.enunciado}` | Correção textual, modificação, testes | Questão | Enunciado da questão |
| `{q.resposta_aluno}` | Correção textual | Questão | Resposta textual enviada pelo aluno |
| `{precisa_saida}` | Correção (modificação) | derivado do enunciado | `SIM`/`NÃO` — se o enunciado pede saída/retorno |
| `{q.codigo_base}` | Correção textual, correção (modificação), testes | Questão | Código anterior do aluno (contexto) |
| `{saida_real}` | Correção textual | Questão (`q.extras["saida_real_codigo_base"]`) | Saída real observada do código-base (evidência objetiva de apoio) |
| `{codigo_aluno_resposta}` | Correção (modificação) | Questão | Código atual do aluno (objeto da avaliação) |
| `{q.tipo}` | Testes | Questão | Tipo da questão |
| `{regra_input}` | Testes | derivado do código | orientação sobre `input()` |
| `{interface}` | Testes | derivado do código | interface atual (quantos/ordem dos inputs) |
| `{regra_entrada}` | Testes | derivado | regras da chave `entrada`/`valores_entrada` |
| `{quantidade}` | Testes | constante | número de testes solicitado (6) |

**Informações do estudante fornecidas à LLM:** na geração, o **código anterior do aluno** e o **enunciado original**; na correção textual, o **enunciado**, o **código-base**, a **evidência objetiva de execução** e a **resposta (texto) do aluno**; na correção de modificação, o **enunciado**, o **código base** e o **código atual do aluno**; na geração de testes, o **enunciado**, o **código base** e a **interface**.

---

## 6. Comparação entre prompts

Tabela comparativa principal:

| Tipo | Etapa | Modelo | Prompt | Variáveis principais | Temperature | Max Tokens | Saída |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Correção | Geração | `qwen/qwen3-vl-4b` | LLM-01 (correção) | `enunciado`, código-base | 0.4 | — | array JSON `[{"tipo","pergunta"}]` |
| Justificativa | Geração | `qwen/qwen3-vl-4b` | LLM-01 (justificativa) | `tipo`, enunciado, código-base | 0.4 | — | array JSON |
| Descritiva | Geração | `qwen/qwen3-vl-4b` | LLM-01 (descritiva) | `tipo`, enunciado, código-base | 0.4 | — | array JSON |
| Modificação | Geração | `qwen/qwen3-vl-4b` | LLM-01 (modificação) | `tipo`, enunciado, código-base | 0.4 | — | array JSON |
| Previsão | Geração | `qwen/qwen3-vl-4b` | LLM-01 (previsão) | `tipo`, enunciado, código-base | 0.4 | — | array JSON `[{"tipo","pergunta","dados_entrada"}]` |
| Correção textual | Correção | `qwen/qwen3-vl-4b` | LLM-02 (compartilhado: descritiva/justificativa/correção textual) | `enunciado`, `codigo_base`, `saida_real`, `resposta_aluno` | 0.15 | 900 | objeto JSON `{nota,status,feedback,acertos,melhorias}` |
| Modificação | Correção | `qwen/qwen3-vl-4b` | LLM-03 (requisitos) | `enunciado`, `precisa_saida`, `codigo_base`, `codigo_aluno_resposta` | 0.15 | 1200 | objeto JSON `{nota,status,requisitos...,feedback}` |
| Geração de testes | Testes | `qwen/qwen3-vl-4b` | LLM-04 | `tipo`, `enunciado`, `codigo_base`, `interface`, `regra_input`, `regra_entrada`, `quantidade` | 0.1 | 1200 | objeto JSON `{testes:[...]}` |

**Principais diferenças:**
- **Geração** usa temperature 0.4 e não envia `max_tokens`; JUSTIFICATIVA, DESCRITIVA e MODIFICAÇÃO compartilham system prompt, cabeçalho, rodapé e formato, diferenciando-se apenas na instrução central. CORREÇÃO e PREVISÃO têm prompts especializados (CORREÇÃO: análise rigorosa que só gera pergunta a partir de erro real; PREVISÃO: inclui o campo `dados_entrada`, convertido deterministicamente em `entradaTestes`).
- **Correção textual** e **modificação** usam temperature 0.15, porém com `max_tokens` distintos (900 vs 1200) e prompts, variáveis e formatos de saída completamente diferentes (nota/feedback de texto vs. requisitos/nota de código).
- A **geração de testes** usa a menor temperatura (0.1) e o formato `{testes:[...]}`.
- O **mesmo modelo** é usado em todas as etapas; o que muda é a tarefa, o prompt, as variáveis de entrada e o formato de saída.

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


