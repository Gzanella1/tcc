# Análise Técnica e Acadêmica — Sistema de Geração de Perguntas e Correção Automática

**Base:** leitura integral/corrente do código-fonte de `gerarPergunta/`, `correcao/`, `outros/tutor.py`, artefatos em `conteudo/` e auditorias `prompts.md`/`relatorio de prompts.md`. Modelo servido localmente via LM Studio (`http://localhost:1234/v1`), `qwen/qwen3-vl-4b`, com API compatível com OpenAI.

---

## 1. Arquitetura geral

O repositório contém **dois módulos operacionais independentes** e um componente auxiliar:

- **`gerarPergunta/`** — módulo *gerador de questões formativas*. Estruturado como pacote (`config/`, `models/`, `services/`, `utils/`) com `main.py` como ponto de entrada.
- **`correcao/`** — módulo *corretor automático*, com pacotes `evaluation/`, `execution/`, `llm/`, `models/`, `parsing/`, `report/`, `tests/`, `utils/` e módulos raiz (`main.py`, `config.py`, `validation.py`).
- **`outros/tutor.py`** — tutor socrático interativo (chat), independente (usa `historico.json`, `MAX_HIST=10`, `TEMPERATURA=0.35`).

Existe ainda **código legado**: `correcao/evaluation/correctors.py` descreve 5 avaliadores mas **não é importado por nenhum outro módulo** (a auditoria `relatorio de prompts.md` o classifica como LLM-05/LLM-06 "NÃO executados", versão anterior arquivada).

**Observação de arquitetura:** a comunicação com o LLM é implementada de **duas formas distintas** — `gerarPergunta/services/ai_client.py` usa o SDK `openai` (`from openai import OpenAI`), enquanto `correcao/llm/client.py` usa `urllib.request` direto. Duplicação de responsabilidade que poderia ser unificada em refatoração futura.

## 2. Fluxo completo do sistema

1. `conteudo/conhecimento.txt` (enunciados + respostas/códigos do aluno) → **KnowledgeLoader** → objetos `Exercicio`.
2. **QuestionOrchestrator** sorteia tipos (**Sorteador**), monta prompt (**QuestionBuilder**), chama o LLM (**AIClient**, com `lru_cache`) → objetos `Pergunta`.
3. **ReportExporter** grava `conteudo/perguntasGeradas.txt` (blocos `Código-base:` + `resposta N -`).
4. O aluno responde; o arquivo passa a conter as respostas.
5. `correcao/main.py` → `carregar_questoes()` (parser com aliases e formatos JSON/texto) → `validar_questao()` (status `pendente`/`erro_entrada`/ok) → dispatcher → estratégia por tipo (execução no sandbox + LLM quando ativo) → **evidências estruturadas** (Etapa 4.3: `{tipo, resumo, dados[, peso]}`) → `report/formatter.py` → `conteudo/correcao.txt` (nota 0–10, status, evidências, feedback).

**Divergência doc×código:** `gp_FLUXO_FUNCIONAMENTO.md` (§10) mostra o relatório com o rótulo "Seu código:", mas `ReportExporter._formatar` realmente usa "Código-base:".

## 3. Análise detalhada — `gerarPergunta/`

| Componente | Status | Papel |
|---|---|---|
| `config/settings.py` | Implementado | LM Studio `localhost:1234/v1`, `MODEL`, `TOTAL_PERGUNTAS=5`, `TEMPERATURE=0.4`, `MAX_TOKENS=512`, `TIPOS_PERGUNTA` (5), caminhos `../conteudo/*`. |
| `models/exercicio.py` | Implementado | `__post_init__` normaliza `titulo`↔`enunciado_original` (alias) e substitui vazio por `"[O ALUNO NÃO ESCREVEU CÓDIGO]"`. |
| `models/pergunta.py` | Implementado | Campos opcionais ficam `None` — não inventa conteúdo. |
| `services/knowledge_loader.py` | Implementado (parcial¹) | `_RE_ENUNCIADO = r"^\s*(\d+)(?:\.\s*|\s+-\s+)(.+?)\s*$"` (aceita "N. Título" legado e "N - Título"), `_RE_RESPOSTA = r"^\s*Resposta\s*(\d+)\s*-\s*$"`. |
| `services/question_builder.py` | Implementado | `_RODAPE` ("Não forneça a resposta", "Não copie o código completo", "Retorne APENAS um JSON válido") + formato `[{"tipo", "pergunta"}]`; 1 método por tipo; `ValueError` para tipo desconhecido. |
| `services/ai_client.py` | Implementado | SDK `openai.OpenAI`; `gerar_pergunta → list[Pergunta]` com rastreabilidade (Fase 3.1); `lru_cache(maxsize=200)`; `_extrair_json` captura primeiro `[`–último `]`. |
| `services/question_orchestrator.py` | Implementado | Cicla exercícios quando há menos do que `total`; retorna pares `(Exercicio, Pergunta)`. |
| `services/report_exporter.py` | Implementado | Cabeçalho "Exercício gerado com base na sua resposta da questão {n}"; bloco `Código-base:`; `resposta {idx} -`. |
| `utils/sorteador.py` | Implementado | Blocos embaralhados de 5 tipos sem repetição dentro da rodada. |
| `main.py` | Implementado | Fluxo loader→orchestrator→exporter com prints no terminal. |

¹ **Imprecisão real:** o `enunciado` capturado é apenas a **primeira linha física** após `N - `; os demais parágrafos de um enunciado multilinha são descartados (caso real em `conteudo/conhecimento.txt`, questão 1). Vale registrar como limitação de ingestão.

## 4. Análise detalhada — `correcao/`

| Componente | Status | Papel |
|---|---|---|
| `config.py` | Implementado | `LLM_MODEL`, `LLM_BASE_URL`, `LLM_TIMEOUT=90`, `USAR_LLM` (`1/0/false/no`), `STATUS_*`, `CONTAINER_IMAGE`, caminhos `../conteudo/perguntasGeradas.txt` e `../conteudo/correcao.txt`. |
| `main.py` | Implementado | `carregar_questoes(ARQUIVO_ENTRADA)`; erro de leitura → `stderr` + exit. |
| `models/questao.py` | Implementado | `Questao` com `extras` (abriga `resposta_formato`, `codigo_base_origem`, etc.). |
| `parsing/parser.py` | Implementado | Aliases `id`/`idx`→id, `tipo`→tipo, `entradaTestes`; formatos JSON e texto; rótulos de código ("Seu código:", "Código-base:", "Código original:", "Código do aluno que originou esta pergunta:"). |
| `validation.py` | Implementado | `_tem_testes_objetivos`, `_pode_gerar_testes`, `_CAMPOS_DO_ALUNO`; pergunta íntegra sem resposta → `pendente` (não erro). |
| `evaluation/dispatcher.py` + `evaluation/strategies/dispatcher.py` | Implementado | Roteamento por tipo; `_resposta_contem_codigo` (só ``` ou label "Código:"), `_extrair_resposta_codigo`. |
| `evaluation/evidencia.py` | Implementado | Contrato Etapa 4.3; fontes `execucao`/`llm`/`heuristica`/`ausente`; default `FONTE_AUSENTE`. |
| `evaluation/correctors.py` | **Legado (morto)** | Não importado por nenhum módulo; contém as chamadas LLM-05/LLM-06. |
| `execution/runner.py` | Implementado | `ExecutionLimits` (frozen dataclass) e `ExecutionResult`; limites de timeout/recurso/bytes; contêiner descartável sem rede; documentação admite "não torna 100% segura". |
| `tests/generator.py` | Implementado | Régua de testes: explícitos do enunciado > gerados por LLM; `_origem` (`enunciado`/`llm`) na Etapa 4.4; ordem real de entrada na Etapa 4.5; anti-autocircularidade. |
| `llm/client.py` | Implementado | `chamar_llm`/`chamar_llm_json` (urllib); falha ou `USAR_LLM=false` → `None`. |
| `report/formatter.py` | Implementado | Formata evidências (`[tipo{peso}]`), trunca a 200 chars, `sim/não` para bool. |
| `utils/text.py` | Implementado | `extrair_assinatura_inputs`, normalização e similaridade de texto. |
| `utils/tipo.py` | Implementado | `normalizar_tipo` (mapa de sinônimos) e `inferir_tipo` (rótulos `[MODIFICACAO]` etc. + palavras-chave). |
| `test_*.py` (13 arquivos) | Implementado | Contratos e regressões (seção 12). |

## 5. Tipos de questão

1. `correcao` — procurar erro lógico/sintático/caso de borda. Avaliação **mista**: `_pergunta_eh_textual` (palavras-chave "qual e o erro", "como corrigir", "explique o erro", "o que esta errado", "por que") → textual; senão execução de código.
2. `justificativa` — explicar uma escolha de implementação → **textual** (reexporta `avaliar` de `texto_llm.py`).
3. `descritiva` — descrever o funcionamento → **textual** (idem).
4. `modificacao` — pensar em uma alteração → **híbrida: 70% testes + 30% LLM**.
5. `previsao` — prever a saída → **determinística, sem LLM** (`Entrada:`/`Saída:`; remove prompts `input()` da saída esperada; fallback legado).

## 6. Prompts

- **LLM-01** (geração, 5 sub-variações): papel do sistema "Você é um tutor de programação." (`ai_client.py:96`) + instrução por tipo + cabeçalho "Exercício {numero}. Enunciado original: ..." + código do aluno + rodapé de regras + formato JSON `[{"tipo", "pergunta"}]`.
- **LLM-02** (correção textual): usa o endpoint da correção; fallback heurístico com `_detectar_conceito_ok` (palavras-chave: minusc/maiusc/lower/upper/vogal, etc.).
- **LLM-03** (requisitos de modificação): avalia conformidade do código do aluno aos requisitos do enunciado; se o LLM falha → mantém apenas a parte objetiva (testes).
- **LLM-04** (geração de testes): acionado apenas quando não há testes explícitos no enunciado; JSON `{testes:[...]}` com valores por variável/prompt; **a saída esperada vem do enunciado, nunca do código do aluno**.
- **LLM-05/LLM-06** — legados (`correctors.py`), **não executados**.

## 7. Cliente LLM

- **`gerarPergunta`**: SDK `openai`, `base_url=http://localhost:1234/v1`, `api_key="lm-studio"`, temperature 0.4, **`max_tokens` não é enviado** (o config guarda 512 "previsto", mas não é passado na chamada); cache LRU de 200 por chave `(numero, titulo, codigo, tipo)`.
- **`correcao`**: `urllib` POST em `{LLM_BASE_URL}/chat/completions`, temperature 0.2 (textual) / 0.15 (modificação), `max_tokens=1200`, timeout 90 s; `USAR_LLM=false` → todos os pontos de LLM viram fallback determinístico.
- **`outros/tutor.py`**: tutor socrático, temperature 0.35, histórico em JSON.

## 8. Estratégias de correção

- `strategies/codigo.py` — executa o código do aluno no sandbox com a régua de testes; detecta erro de execução (`_detectar_erro_execucao`) com motivo e classificação; fonte predominante `execucao`.
- `strategies/previsao.py` — pares `Entrada:`/`Saída:` (prioridade 1); blocos mistos apenas na ausência de marcadores; fallback legado; saída comparada por similaridade após normalização.
- `strategies/texto_llm.py` — LLM com fallback heurístico de conceitos (rótulos `TIPO_HEURISTICA`/`TIPO_LLM`).
- `strategies/modificacao.py` — híbrido: parte objetiva (testes) com peso 0.7 + LLM (requisitos) com peso 0.3; sem testes → só LLM; código-placeholder → `ausente`; LLM falha → apenas parte objetiva.
- `strategies/correcao.py` — rota textual ou por execução, conforme `_pergunta_eh_textual`.
- `strategies/descritiva.py` e `strategies/justificativa.py` — reexportam `avaliar` de `texto_llm.py`.

## 9. Correção híbrida 70/30 (questões de modificação)

`nota = 0.7·A + 0.3·B`, onde:

- `A` = fração de testes que o código do aluno passa na **execução real no sandbox**, com stdin remontado deterministicamente (`_montar_entrada_por_assinatura`, na ordem da interface real extraída do código via `extrair_assinatura_inputs`);
- `B` = conformidade semântica do código com os requisitos do enunciado, avaliada pelo LLM.

A evidência objetiva carrega `peso=0.7` (confirmado em `test_procedencia_testes.py`). Esse design garante que, mesmo com o LLM indisponível, a nota objetiva cobre o essencial da questão.

## 10. Modelos e configuração

- Único modelo: `qwen/qwen3-vl-4b` (local, LM Studio), usado **na geração e na correção**; substituível por variáveis de ambiente (`LLM_MODEL`, `LLM_BASE_URL`).
- Sem dependência de nuvem: privacidade, custo zero e latência local; porém modelo pequeno (4 B) → sensível a formato JSON e a sutilezas de erro.
- Temperaturas: geração 0.4; correção textual 0.2; requisitos de modificação 0.15; geração de testes 0.1; tutor 0.35.

## 11. Erros e fallbacks

- JSON inválido (geração) → pergunta descartada, lista vazia — o fluxo não quebra.
- LLM indisponível → retorna `None`, propagado como evidência `ausente` e fallback heurístico/objetivo.
- Sem código localizável → feedback "Não foi possível localizar o código da questão para calcular a saída." e evidência com fonte `ausente` (nota 0, não é erro de entrada).
- Runtime de contêiner indisponível → evidência com erro "Runtime de container indisponível".
- Código muito grande (ex.: 20 000 linhas) → `limit_exceeded`.
- Resposta íntegra sem avaliação → fica `pendente` (não `erro_entrada`).
- **Anomalia de qualidade:** `utils/text.py`, na função `classificar_erro_execucao`, contém um caractere CJK solto ("Esta função仅 identifica…") — resíduo de edição.
- **Anomalia de processo:** teste morto em `test_parser_contrato_geracao.py:128-141` (ver seção 12).

## 12. Testes

- **`gerarPergunta/test_contrato_geracao.py`** — contrato da Fase 3.1 (loader, pares `(Exercicio, Pergunta)`, cache, JSON, 5 tipos) com `FIXTURE_CONHECIMENTO`.
- **`correcao/`** — 13 arquivos `test_*.py`:
  - `test_parser_validation` — schema canônico (`id:10`, `entradaTestes`/`saida_esperada`), nega `q.codigo`/`q.saida`.
  - `test_parser_contrato_geracao` — formatos de bloco, resposta futura separada da origem. **Teste morto confirmado:** a linha 128 abre um `"""` que transforma `test_arquivo_real_perguntas_geradas_carrega_5_questoes` (linhas 129–141) em string; o arquivo real passa a ser validado pela função solta `test_arquivo_real_perguntas_geradas_carrega_questoes_validas` (≥ 1 questão no lugar de 5 fixas).
  - `test_procedencia_testes` — `_origem` (`enunciado`/`llm`), `testes_por_origem`, soma X+Y, peso 0.7, sem vazamento de `_origem` para testes executados/relatório.
  - `test_runner_security` — contêiner, sintaxe, `limit_exceeded`, mensagens de erro.
  - `test_ordem_inputs_gerados` — 9 casos de remontagem determinística de stdin (ordem da interface > ordem escolhida pelo LLM).
  - `test_geracao_interface_aluno`, `test_incompatibilidade_entrada`, `test_interface_dinamica`, `test_evidencia_rastreabilidade`, `test_evidencia_conteudo`, `test_evidencia_propagacao` — regressões das Etapas 4.3–4.5.
- Suite executada com `unittest`. Último commit: `16aba48 codigo com erro nos casos de teste` — verificar se os "casos de teste" quebrados coincidem com o teste morto relatado.

## 13. Análise acadêmica

Pedagogicamente, o sistema opera como **avaliação formativa e tutor socrático**: transforma o código real do aluno em ocasião de reflexão — prever saída, justificar escolha, descrever funcionamento, localizar erro e propor modificação — estimulando a **metacognição** sobre a própria produção.

Tecnicamente, o diferencial é o **design híbrido LLM + núcleo determinístico**: execução, sintaxe, similaridade, pesos e deduplicação são objetivos; o LLM atua apenas como componente semântico, com múltiplos fallbacks. A robustez é reforçada por **rastreabilidade** (Etapas 4.3/4.4: evidências `{tipo, resumo, dados, peso}` e proveniência da régua via `_origem`) e por **anti-autocircularidade** (a resposta do aluno nunca gera a própria régua de avaliação; a saída esperada vem do enunciado, não do código do aluno).

**Limitações a reconhecer:** ingestão de enunciados multilinha truncada; dependência de JSON válido do modelo 4B; duplicação de clientes LLM; código morto (`evaluation/correctors.py`); teste morto em `test_parser_contrato_geracao.py`.

## 14. Contribuição

1. Pipeline completo e auditável: conhecimento → pergunta personalizada → resposta → correção com feedback baseado em evidência.
2. Correção híbrida com ponderação explícita (70/30) e proveniência da régua de testes.
3. Sandbox seguro (execução em contêiner sem rede) com limites de recurso.
4. Execução 100% local (LM Studio) — viável em cenário institucional sem envio de dados à nuvem.
5. Avaliação textual resiliente: fallback heurístico quando o LLM está indisponível.

## 15. Relação com a pergunta de pesquisa

A pergunta-motriz ("como um LLM pode atuar como tutor formativo em programação?") é respondida na implementação: **intervenção pedagógica personalizada a partir do código do estudante**, com correção **confiável** (núcleo determinístico) e **explicável** (evidências estruturadas com origem e peso) — não um simples "chat avaliador".

## 16. Material para a Metodologia

- Citar: `gerarPergunta/services/*`, `models/*`, `config/settings.py`; `correcao/parsing/parser.py`, `validation.py`, `execution/runner.py`, `evaluation/strategies/*`, `evaluation/evidencia.py`, `tests/generator.py`, `llm/client.py`, `report/formatter.py`, `config.py`; artefatos `conteudo/{conhecimento,perguntasGeradas,correcao}.txt`; auditorias `prompts.md` e `relatorio de prompts.md`.
- Resultado real de referência: `conteudo/correcao.txt` (30/08/2026) — 1 questão, média 0.00/10, status `erro`, evidência `[llm]` "Avaliação via LLM (nota original 0.00)", `piso_aplicado: não`.
- Classificar cada componente (implementado/parcial/legado) conforme as seções 3 e 4.

## 17. Estrutura sugerida do capítulo de desenvolvimento

1. Visão geral da arquitetura (módulos e responsabilidades).
2. Geração de perguntas (entrada, sorteio, prompt, LLM, relatório).
3. Correção por tipo (estratégias e fluxos).
4. Correção de código: sandbox, régua de testes e proveniência.
5. Correção híbrida 70/30 com demonstração numérica.
6. Rastreabilidade e evidências (Etapas 4.3–4.5).
7. Testes e limitações (incluindo as anomalias encontradas).

## 18. Tabela final de mapeamento

| Peça | Status | Arquivo |
|---|---|---|
| Geração de perguntas | implementado | `gerarPergunta/services/*`, `models/*` |
| Sorteio de tipos | implementado | `gerarPergunta/utils/sorteador.py` |
| Cliente LLM (SDK openai) | implementado | `gerarPergunta/services/ai_client.py` |
| Parser e validação | implementado | `correcao/parsing/parser.py`, `correcao/validation.py` |
| Sandbox de execução | implementado | `correcao/execution/runner.py` |
| Estratégia código / previsão | implementado | `correcao/evaluation/strategies/{codigo,previsao}.py` |
| Estratégia textual híbrida | implementado | `correcao/evaluation/strategies/texto_llm.py` |
| Modificação híbrida 70/30 | implementado | `correcao/evaluation/strategies/modificacao.py` |
| Geração/proveniência de testes | implementado | `correcao/tests/generator.py`, `test_procedencia_testes.py` |
| Evidências e feedback | implementado | `correcao/evaluation/evidencia.py`, `report/formatter.py` |
| Avaliadores antigos | **legado/morto** | `correcao/evaluation/correctors.py` |
| Teste de 5 questões do arquivo real | **morto (`"""`)** | `correcao/test_parser_contrato_geracao.py:128–141` |
| Ingestão de enunciado multilinha | **parcial** | `gerarPergunta/services/knowledge_loader.py` |
| `MAX_TOKENS` na geração | **não aplicado** | `gerarPergunta/config/settings.py` vs `ai_client.py` |