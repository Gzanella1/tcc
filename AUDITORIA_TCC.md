# AUDITORIA TÉCNICA E METODOLÓGICA DO PROJETO DE TCC

**Data da auditoria:** 07/09/2026
**Escopo:** MÓDULO 1 (casos de teste e execução de código), MÓDULO 2 (geração de perguntas), MÓDULO 3 (correção das respostas).
**Método:** inspeção do código-fonte (`correcao/`, `gerarPergunta/`), dados reais de execução (`conteudo/*.txt`), documentação interna (READMEs, `EXPLICACAO_GERAR_PERGUNTAS.md`, `gp_EXPLICACAO_CODIGO.md`, `gp_FLUXO_FUNCIONAMENTO.md`, `exemplosPGeCorrecao/prompts.md`), execução das suítes de teste do próprio sistema (212 testes automatizados) e observação do ambiente (LLM ativo em `localhost:1234`, Docker disponível).

**Convenções de classificação:**
- `[DEFINIDO]` — implementado no código e/ou documentado.
- `[PARCIAL]` — há implementação, mas com restrição ou propósito incompleto.
- `[NÃO DEFINIDO]` — ausente no código e na documentação.
- `[DECISÃO DO AUTOR]` — escolha arbitrada do autor (não é lacuna; é posicionamento metodológico).
- `[DIVERGÊNCIA ENTRE DOCUMENTAÇÃO E IMPLEMENTAÇÃO]` — documentação divergente do código.

Todas as informações abaixo foram verificadas no código ou nos arquivos de execução. Nenhum parâmetro, quantidade ou resultado foi inventado.

---

## 1. RESUMO EXECUTIVO

O projeto implementa um tutor de programação com IA em dois módulos encadeados: `gerarPergunta` produz perguntas personalizadas a partir do enunciado e do código anterior do aluno (`codigo_base`), e `correcao` avalia as respostas com estratégias híbridas (execução isolada em container + avaliação semântica via LLM). O modelo único `qwen/qwen3-vl-4b` serve a todas as tarefas (geração, correção, geração de testes) por meio de LM Studio (`http://localhost:1234/v1`).

**Estado geral: bom e verificável.** Os três módulos estão implementados de ponta a ponta, com isolamento operacional seguro da execução (container descartável, sem rede, sem escrita, com limites rígidos), rastreabilidade estruturada da origem da nota (evidências `execucao`/`llm`/`heuristica`/`ausente`) e procedência da régua de testes (`enunciado`/`llm`). As 212 suítes de teste do próprio sistema passam no ambiente atual (185 na correção, incluindo integração real com Docker; 27 na geração).

**Avaliação por módulo:**
- MÓDULO 1 — forte: geração/validação/deduplicação de testes com origem rastreada e execução segura. Lacuna principal: contingência para ausência de runtime (sem fallback para processo local); documentação interna desatualizada sobre o formato de entrada/saída.
- MÓDULO 2 — funcional, com uma fragilidade evidente nos dados reais: a última execução gerou apenas 4 perguntas (faltou PREVISAO) e 3 com enunciado vazio, e `MAX_TOKENS` está declarado mas não é enviado à API.
- MÓDULO 3 — robusto: 5 tipos de estratégia, nota sempre limitada/calculada pelo código, piso heurístico em resposta textual e pesos 70/30 na modificação.

**Divergências mais relevantes (detalhe na seção 13):** documentação interna da geração desatualizada quanto ao formato de títulos e de saída; e o arquivo de questões corrente (`perguntasGeradas.txt`, 4 itens) não corresponde à quantidade de itens corrigidos no relatório corrente (`correcao.txt`, 5 itens) — os arquivos reais vêm de execuções diferentes.

---

## 2. MÓDULO 1 — CASOS DE TESTE E EXECUÇÃO DE CÓDIGO

### 2.1 Geração e obtenção de casos de teste

| Item | Classificação | Informação |
| --- | --- | --- |
| Origem dos testes | `[DEFINIDO]` | `correcao/tests/generator.py`: `obter_testes()` combina testes declarados no enunciado/campo (`obter_testes_explicitos`, origem `enunciado`) com testes gerados por LLM (`gerar_testes_com_llm`, origem `llm`). |
| Quantidade-alvo | `[DEFINIDO]` | `TESTES_ALVO = 6` (`correcao/config.py:30`). |
| Geração via LLM | `[DEFINIDO]` | `_gerar_testes_llm_once` com até **3 tentativas**; `temperature=0.1`, `max_tokens=1200` (`generator.py:551-552`). Só acionada quando o código usa `input()`. |
| Validação e deduplicação | `[DEFINIDO]` | `validar_testes()` (exige entrada/saída válidas; `requer_input`) e `deduplicar_testes()`; colisões por entrada. |
| Interface de inputs | `[DEFINIDO]` | `_codigo_para_interface()` extrai assinatura dos `input()` (ordem/quantidade); `_montar_entrada_por_assinatura`, `_normalizar_entrada_dinamica`, `limpar_entrada_interativa`; incompatibilidade de entrada detectada (`entrada não atende interface`). |
| Procedência/rastreio | `[DEFINIDO]` | Marcador interno `_CHAVE_ORIGEM = "_origem"` (`ORIGEM_ENUNCIADO="enunciado"`, `ORIGEM_LLM="llm"`); a chave interna não vaza para a Questão e a contagem é exposta na evidência `testes_por_origem`. |
| Régua não autocircular | `[DEFINIDO]` | A saída esperada é definida pelo **enunciado** (fonte primária); o `codigo_base` é "apenas contexto; NÃO é gabarito e NÃO define a saída esperada". O código avaliado (`codigo_aluno_resposta`) nunca contribui para gerar a própria régua. |

### 2.2 Execução de código (isolamento e segurança)

| Item | Classificação | Informação |
| --- | --- | --- |
| Backend de execução | `[DEFINIDO]` | `EXEC_BACKEND = "container"` (`config.py:33`); runtime auto (docker/podman). No ambiente auditado: **Docker presente e funcional**; imagem `python:3.12-alpine` disponível. |
| Isolamento do container | `[DEFINIDO]` | `execution/runner.py`: `--rm --network none --read-only --user 65534:65534` + `tmpfs`/working dir para o código; **sem montagem do projeto**, sem rede e sem escrita persistente. Confirmado por integração: `test_tentativa_de_acesso_rede_falha` e `test_tentativa_de_acessar_arquivo_do_host_falha` passam. |
| Limites de recursos | `[DEFINIDO]` | Timeout `3s`, CPU `3s`, memória `128MB`, processos `32`, arquivos abertos `64`; limites de bytes: código/stdin/stdout/stderr `65536` cada (`config.py:37-45`). |
| Desfechos controlados | `[DEFINIDO]` | `ExecutionResult` cobre `timeout`, `erro_execucao`, `returncode`, `stdout/stderr` truncados (flags `stdout_truncated`/`stderr_truncated`), `limit_exceeded`. |
| Verificação prévia | `[DEFINIDO]` | Checagem de sintaxe via `ast`; código acima do limite de bytes e entrada acima do limite não executam (`test_codigo_grande_nao_executa`, `test_entrada_grande_nao_executa`). |
| Contingência sem runtime | `[PARCIAL]` | Sem container disponível, a execução retorna desfecho de erro (`test_container_indisponivel`) — **não há fallback para processo/subprocesso local**. Decisão aceitável para o TCC, mas registrada como limitação. |

### 2.3 Comparação de saídas

| Item | Classificação | Informação |
| --- | --- | --- |
| Limiares | `[DEFINIDO]` | `LIMIAR_EXATO = 0.99`, `LIMIAR_APROX = 0.90` (`config.py:26-27`). |
| Classes de resultado | `[DEFINIDO]` | Passou / Falhou / Incompatível (teste descartado quando a entrada não atende a interface), com `motivo` detalhado (ex.: "saída esperada presente; há saída extra", "Processo retornou código diferente de zero"). |
| Similaridade | `[DEFINIDO]` | `difflib.SequenceMatcher` para comparação textual das saídas; evidência registra a similaridade (`execucao`, `referencia=testes_executados`). |

**Conclusão do módulo:** `[DEFINIDO]` na totalidade dos itens operacionais; `[PARCIAL]` apenas na contingência de runtime.

---

## 3. MÓDULO 2 — GERAÇÃO DE PERGUNTAS

### 3.1 Contexto pedagógico e entrada

| Item | Classificação | Informação |
| --- | --- | --- |
| Fonte de conhecimento | `[DEFINIDO]` | `conteudo/conhecimento.txt` com 5 questões (Caracteres únicos, Diferença simétrica, Busca reversa, Ordem crescente, Ordem decrescente) e blocos `resposta N -`. Carregado por `knowledge_loader.py` como `Exercicio(numero, enunciado_original, codigo_base)`; alias `titulo ↔ enunciado_original` sincronizado. |
| Separador enunciado | `[DEFINIDO]` | Regex aceita `"N. "` e `"N - "` (`_RE_ENUNCIADO = ^\s*(\d+)(?:\.\s*|\s+-\s+)(.+?)\s*$`). Enunciados só são reconhecidos fora dos blocos de resposta (nunca captura código como título). |
| Enunciado original preservado | `[DEFINIDO]` | Mantido separado do `codigo_base`; exportado para a correção via `Questao.extras["enunciado_origem"]` (campo sem campo canônico, permanece em extras). |

### 3.2 Estratégia de personalização

| Item | Classificação | Informação |
| --- | --- | --- |
| Tipos | `[DEFINIDO]` | `TIPOS_PERGUNTA = [correcao, justificativa, descritiva, modificacao, previsao]`; `TOTAL_PERGUNTAS = 5` (`settings.py`). |
| Seleção de exercícios | `[DEFINIDO]` | `question_orchestrator.py`: slots `exercicios[:total]`; se houver menos exercícios que perguntas, repete em ciclo (`exercicios[i % qtd]`). Lista vazia → `ValueError`. |
| Sorteio de tipos | `[DEFINIDO]` | `Sorteador`: embaralha bloco completo (`random.shuffle`) — nenhum tipo repete antes de esgotar os 5; corta no tamanho pedido. **Sem seed** → ordem varia a cada execução. |
| Prompts por tipo | `[DEFINIDO]` | `question_builder.py`: JUSTIFICATIVA/DESCRITIVA/MODIFICAÇÃO usam cabeçalho+rodapé comuns; CORREÇÃO tem prompt especializado (regra "NUNCA invente um erro"; análise rigorosa de sintaxe/lógica/borda); PREVISÃO tem rodapé próprio exigindo `dados_entrada` estruturado. Rodapé comum: "Não forneça a resposta", "Não copie o código completo", JSON obrigatório. |

### 3.3 Chamada ao LLM e transformação

| Item | Classificação | Informação |
| --- | --- | --- |
| Modelo/servidor | `[DEFINIDO]` | `MODEL = qwen/qwen3-vl-4b`; `LM_STUDIO_BASE_URL = http://localhost:1234/v1`; `api_key = lm-studio`; biblioteca `openai`. Servidor **ativo** no momento da auditoria (modelo listado em `/v1/models`). |
| Parâmetros de geração | `[PARCIAL]` | `TEMPERATURE = 0.4` é enviado; `MAX_TOKENS = 512` está **declarado mas não é passado** na chamada `chat.completions.create`. |
| Cache | `[DEFINIDO]` | `AIClient` usa `functools.lru_cache(200)` em `_chamar_api_cached`, chave `(numero, titulo, codigo_base, tipo)` — em memória, por execução. |
| Conversão da resposta | `[DEFINIDO]` | Extrai array JSON; cada item vira `Pergunta(tipo, pergunta, exercicio_numero, enunciado_origem)`; campos opcionais (`conceito_avaliado`, `evidencia_codigo`, `resposta_esperada`, `rubrica`) permanecem `None` — **o sistema não inventa evidência**. JSON inválido → slot vazio (aviso). |
| PREVISÃO (dados de entrada) | `[DEFINIDO]` | `services/dados_entrada.py`: transformação **determinística** `dados_entrada` (dict/list semântico) → `entrada_testes` (string stdin) por serializadores (lista linear, mapa com dict aninhado, dict de escalares; formato desconhecido rejeitado); **validação por execução de referência** (`executar_referencia`, timeout 5s): entrada vazia, código que falha, timeout ou sem saída → PREVISÃO rejeitada na geração. Exportada como `entradaTestes: "..."` em linha única. |

### 3.4 Validação da qualidade das perguntas geradas

| Item | Classificação | Informação |
| --- | --- | --- |
| Validação estrutural (JSON) | `[DEFINIDO]` | Apenas o formato JSON e presença de campo `pergunta` não vazio. |
| Validação semântica/pedagógica | `[NÃO DEFINIDO]` | Não há camada pós-geração que valide qualidade pedagógica, aderência do tipo retornado ao tipo pedido, ou ausência de "resposta escondida". Reconhecido na documentação interna ("O modelo não é validado semanticamente"). |
| Robustez a falha da IA | `[PARCIAL]` | Falha de JSON → slot vazio (sem retry na geração de questões); o arquivo final pode sair com menos perguntas que `TOTAL_PERGUNTAS`. **Observado nos dados reais**: `perguntasGeradas.txt` atual tem **4** perguntas (faltou PREVISAO) e **3 com enunciado vazio** (CORRECAO, JUSTIFICATIVA, DESCRITIVA). |

**Conclusão do módulo:** funcional e rastreável; fragilidades em `MAX_TOKENS` (não aplicado), ausência de validação semântica pós-geração e qualidade instável da saída real.

---

## 4. MÓDULO 3 — CORREÇÃO DAS RESPOSTAS

### 4.1 Leitura e parsing

| Item | Classificação | Informação |
| --- | --- | --- |
| Formatos de entrada | `[DEFINIDO]` | `parsing/parser.py`: tenta JSON primeiro (com aliases de campos); fallback para blocos de texto com separadores `=====`. |
| Contrato canônico | `[DEFINIDO]` | `codigo_base` = código ANTERIOR do aluno (contexto; rótulos "Seu código:", "Código-base:", "Código original:", origem da geração); `codigo_aluno_resposta` = NOVO código da resposta (região de resposta, cercas ```/separador `----`), **nunca alimenta codigo_base**; `entradaTestes` = entrada obrigatória de PREVISAO, lida fora da região de resposta; `dados_entrada` (intermediário da geração) **nunca persiste** em Questao/extras. |
| Campos intermediários | `[DEFINIDO]` | `_CAMPOS_INTERMEDIARIOS_GERACAO` filtra `dados_entrada` na normalização JSON. |
| Detecção de pergunta textual | `[DEFINIDO]` | `_enunciado_pede_texto()` (descritiva/justificativa/correção com "qual é o erro?", "por que", "explique", etc.) e `_formato_resposta()`. |

### 4.2 Validação de integridade

| Item | Classificação | Informação |
| --- | --- | --- |
| Validação de campos | `[DEFINIDO]` | `validation.py`: separa questão íntegra, campos ausentes essenciais (`STATUS_ERRO`), pergunta íntegra sem resposta do aluno (`STATUS_PENDENTE` — "aguardando resposta do aluno; nada foi avaliado") e inconclusivos (`STATUS_INCONCLUSIVO`). |
| Normalização de tipo | `[DEFINIDO]` | `utils/tipo.py`: `normalizar_tipo()` (mapeia sinônimos: corrigir/alterar/prever/...) e `inferir_tipo()` por rótulos `[TIPO]` ou palavras-chave. `dispatcher.py` roteia por tipo com fallback para `codigo`→`texto_llm`. |

### 4.3 Estratégias de avaliação (nota, status, feedback)

| Tipo | Classificação | Estratégia |
| --- | --- | --- |
| `previsao` | `[DEFINIDO]` | **Determinística, sem LLM.** Executa o `codigo_base`; modo pares (similaridade 0.99 via `LIMIAR_EXATO`) e modo legado (0.90 via `LIMIAR_APROX`, quando a entrada vem de `entradaTestes`). Evidência `[execucao]`, `modo: legado`, `referencia: testes_executados`. Na execução real: nota 10.00, similaridade 1.000. |
| `correcao` textual | `[DEFINIDO]` | Se `_pergunta_eh_textual` → executa `codigo_base` sem `input()` para obter `saida_real_codigo_base` e delega a `texto_llm`. Piso objetivo: se conceito central (maiúsculas/minúsculas/padronização/vogais) detectável na resposta → nota mínima **7.0** (status elevado de `erro`→`parcial`). |
| `correcao` código | `[DEFINIDO]` | Com testes: execução determinística contra a régua (`avaliar_codigo`). Sem testes: delega a `texto_llm`. |
| `modificacao` | `[DEFINIDO]` | Com testes: `nota = 0.7×testes + 0.3×nota_llm(requisitos)`; sem testes: `nota = nota_llm`; LLM indisponível: parte objetiva (testes). `precisa_saida` derivado do enunciado; LLM retorna `requisitos_identificados/atendidos/faltantes`. |
| `justificativa` / `descritiva` | `[DEFINIDO]` | Reexportam `avaliar` de `texto_llm` (`justificativa.py`, `descritiva.py`) — mesma estratégia textual. |
| Tipo desconhecido | `[DEFINIDO]` | Fallback: se há código de resposta → `codigo`; senão `texto_llm`. |
| Nota final | `[DEFINIDO]` | Sempre limitada a `[0,10]` e arredondada pelo código (`max(0,min(10,round(nota,2)))`); ponderação 70/30 também via código. Status normalizado (`ok|parcial|erro`). |

### 4.4 Feedback e rastreabilidade

| Item | Classificação | Informação |
| --- | --- | --- |
| Feedback | `[DEFINIDO]` | Campo `feedback` (curto) + `acertos`/`melhorias` (textual) ou `requisitos_atendidos/faltantes` (modificação) listados em Detalhes. |
| Evidências estruturadas | `[DEFINIDO]` | `evaluation/evidencia.py`: contrato `{tipo, resumo, dados, peso}`; fontes canônicas `FONTE_EXECUCAO="execucao"`, `FONTE_LLM="llm"`, `FONTE_HEURISTICA="heuristica"`, `FONTE_AUSENTE="ausente"` e `normalizar_fonte()`. Transportadas por `Resultado` sem quebrar o contrato. |
| Relatório | `[DEFINIDO]` | `report/formatter.py`: `CORREÇÃO AUTOMÁTICA`, data, total, média geral, aprovadas (nota >= 7.0), e por questão: nota, status, feedback, enunciado, detalhes, evidências e testes executados (status PASSOU/FALHOU/INCOMPATÍVEL + entradas/saídas/motivo/stderr). |
| Parâmetros LLM na correção | `[DEFINIDO]` | `texto_llm`: temperature 0.15, max_tokens 900. `modificacao`: temperature 0.15, max_tokens 1200. `llm/client.py` defaults 0.2/1200 (não usados nos fluxos atuais). `LLM_TIMEOUT = 90`, `USAR_LLM` default "1". |

**Conclusão do módulo:** `[DEFINIDO]` na essência; a única decisão de contorno é o piso heurístico (seção 12).

---

## 5. INFORMAÇÕES COMPARTILHADAS

| Item | Classificação | Informação |
| --- | --- | --- |
| Modelo único | `[DEFINIDO]` | `qwen/qwen3-vl-4b` no mesmo LM Studio para geração, correção textual, modificação e geração de testes. |
| Fluxo final | `[DEFINIDO]` | `conhecimento.txt` → `gerarPergunta` → `perguntasGeradas.txt` → (respostas do aluno adicionadas) → `correcao` → `correcao.txt`. |
| Limites de execução compartilhados | `[DEFINIDO]` | O mesmo `runner` executa o `codigo_base` na correção (evidência `saida_real`) e a validação de referência da PREVISÃO na geração. |
| Rastreabilidade cruzada | `[DEFINIDO]` | `enunciado_origem` nasce na geração e chega à correção via extras; `codigo_base` cruza a fronteira intacto; `entradaTestes` (PREVISAO) gerada deterministicamente e consumida pela correção. |

---

## 6. REPRODUTIBILIDADE

- `[x] DEFINIDO` — Ambiente: LM Studio ativo com o modelo `qwen/qwen3-vl-4b` em `http://localhost:1234/v1`; Docker disponível com imagem `python:3.12-alpine`; Python 3 disponível (venv presente).
- `[x] DEFINIDO` — Configuração por variáveis de ambiente com padrões válidos (`ARQUIVO_ENTRADA`, `ARQUIVO_SAIDA`, `LLM_BASE_URL`, `LLM_MODEL`, `USAR_LLM`, `LLM_TIMEOUT`, `EXEC_*`).
- `[x] DEFINIDO` — Caminhos padrão coerentes: `../conteudo/conhecimento.txt`, `../conteudo/perguntasGeradas.txt`, `../conteudo/correcao.txt`.
- `[x] PARCIAL` — Dependências: declaradas apenas em documentação (README/`EXPLICACAO`); **não foi localizado `requirements.txt`/`pyproject.toml`** para a instalação reproduzível (a dependência real é apenas a biblioteca `openai`, usada pela geração; correção usa stdlib).
- `[x] PARCIAL` — Pipeline executado manualmente em etapas (gerar → editar respostas → corrigir); não há um script único ponta-a-ponta.
- `[x] PARCIAL` — Ausência de `seed`: sorteio de tipos e geração dependem de aleatoriedade/LLM, tornando saídas não determinísticas entre execuções.
- `[x] DEFINIDO` — Verificação objetiva no momento da auditoria: **212/212** testes automatizados passam (`correcao`: 185, incluindo integração Docker real; `gerarPergunta`: 27).

---

## 7. CRITÉRIOS DE QUALIDADE DAS QUESTÕES

| Critério | Classificação | Estado |
| --- | --- | --- |
| Pergunta não entrega a resposta | `[DEFINIDO]` | Regras obrigatórias no rodapé dos prompts da geração ("Não forneça a resposta", "Não copie o código completo"). |
| Pergunta fundamentada em erro real | `[DEFINIDO]` | Prompt de CORREÇÃO exige erro real, verificável e demonstrável; proíbe inventar erro; exige tentativa de refutação. |
| Validação semântica da pergunta gerada | `[NÃO DEFINIDO]` | Não há camada de qualidade pós-geração (nem aderência de tipo, nem detecção de resposta espontânea). |
| Qualidade observada na saída real | `[PARCIAL]` | `perguntasGeradas.txt` atual: 4 perguntas (não 5), 3 com enunciado vazio; indica instabilidade da execução real (falha de JSON/previsão rejeitada). |
| Pergunta não respondida | `[DEFINIDO]` | Correção marca `PENDENTE` (não penaliza, não avalia). |
| Fronteira contexto vs. gabarito | `[DEFINIDO]` | `codigo_base` é contexto/apoio e nunca gabarito; regra repetida nos prompts de correção, modificação e testes. |

---

## 8. AVALIADORES

| Avaliador | Onde | Participação do LLM na nota |
| --- | --- | --- |
| Execução determinística (container) | `execution/runner.py` | Não participa. Régua de testes pode vir da LLM (origem `llm`), mas a nota é derivada da execução. |
| `previsao` | `strategies/previsao.py` | **0%** — nota 100% pela comparação determinística de saída. |
| `texto_llm` (justificativa/descritiva/correção textual) | `strategies/texto_llm.py` | **100%** (campo `nota`), com piso objetivo de 7.0 e fallback heurístico 7.5/3.0. |
| `modificacao` | `strategies/modificacao.py` | **30%** (requisitos) + **70%** testes; sem testes → 100% LLM; LLM falhou → 100% parte objetiva. |
| `correcao` (código + testes) | `strategies/correcao.py` | **0% na nota**; LLM só na régua (geração de testes). |
| Heurística local | `texto_llm` (fallback) | 100% heurística quando o LLM não retorna JSON válido. |

Regra transversal: **a decisão final é sempre do código** — valida, limita `[0,10]`, arredonda e pondera; a LLM fornece a nota semântica.

---

## 9. MÉTRICAS

| Métrica | Onde é calculada | Informação |
| --- | --- | --- |
| Nota final (0–10) | `Resultado.nota` | Arredondada e limitada pelo código em todas as estratégias. |
| Média geral | `report/formatter.py` | `sum(notas)/len(resultados)`. |
| Aprovação | `report/formatter.py` | Contagem de `nota >= 7.0`. |
| Similaridade | `previsao`/`codigo` | `difflib.SequenceMatcher`; limiares 0.99 (exato) e 0.90 (aprox.). |
| Procedência dos testes | `generator.py` | `testes_por_origem` (`enunciado` vs `llm`) exposto em evidência. |
| Nota original vs. piso | `texto_llm` | `nota_original` registrada na evidência antes do piso; `piso_aplicado: sim/não`. |
| Nota composta | `modificacao` | `0.7×testes + 0.3×LLM`, com clamps. |
| Desfechos de execução | `runner.py` | `timeout`, `erro_execucao`, truncamentos, `returncode`. |

**Resultados reais observados** (`conteudo/correcao.txt`, 07/09/2026 19:55): 5 questões — justificativa 0.00 (erro), previsão 10.00 (ok), modificação 8.83 (parcial, 5/6 testes, `testes_por_origem={'enunciado':0,'llm':6}`), correção 10.00 (ok), descritiva 10.00 (ok). Média 7.77, 4 aprovadas. Evidências `[llm]` (com nota original e `piso_aplicado: não`) e `[execucao]` presentes no relatório.

---

## 10. PROCEDIMENTO EXPERIMENTAL

| Item | Classificação | Informação |
| --- | --- | --- |
| Instalação das 5 questões | `[DEFINIDO]` | `conteudo/conhecimento.txt`: questões de Algoritmos (Caracteres únicos, Diferença simétrica, Busca reversa, Ordem crescente, Ordem decrescente). |
| Geração | `[DEFINIDO]` | Uma execução de `gerarPergunta` (`TOTAL_PERGUNTAS=5`, temperatura 0.4). Execução observada em `exemplosPGeCorrecao` e `conteudo/perguntasGeradas.txt`. |
| Respostas dos alunos | `[DEFINIDO]` | Preenchidas manualmente em `resposta N -` (ex.: `a.txt`, `pergunttaBaseREspondida(...).md`). Vazio → `PENDENTE`. |
| Correção | `[DEFINIDO]` | Uma execução de `correcao` com `USAR_LLM=1`, LLM ativa, Docker ativo, limiares 0.99/0.90, `TESTES_ALVO=6`. |
| Testes automatizados do sistema | `[DEFINIDO]` | Executados na auditoria: 212/212 passam em ~82s. |
| Reexecução do experimento pelo autor | `[NÃO DEFINIDO]` | Não há roteiro formal gravado (ordem de comandos/arquivos a editar) nem dados das execuções anteriores preservados. |

---

## 11. MATRIZ FINAL DE LACUNAS

| Módulo | Item | Status | Informação | Evidência | Arquivo | Decisão |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Fallback de execução sem container | `[PARCIAL]` | Sem runtime, não há fallback local; resposta vira erro. | `test_container_indisponivel` | `execution/runner.py` | Aceitar e registrar no TCC, ou adicionar fallback |
| 1 | Documentação do formato de entrada/saída | `[DIVERGÊNCIA]` | Doc da geração descreve formato antigo (ver seção 13). | READMEs/EXPLICACAO vs código | `gerarPergunta/` | Atualizar docs |
| 2 | `MAX_TOKENS` não aplicado | `[DIVERGÊNCIA]` | Declarado 512; chamada não envia `max_tokens`. | `settings.py` vs `ai_client.py` | `gerarPergunta/config/settings.py` | Enviar `max_tokens` na chamada |
| 2 | Validação semântica da pergunta | `[NÃO DEFINIDO]` | Sem camada pós-geração (qualidade/aderência de tipo). | comentários e prompt | `services/question_builder.py` | Decidir escopo no TCC |
| 2 | Fuzzy varias por falha JSON | `[PARCIAL]` | Slot vazio sem retry; saída real com 4/5 perguntas e 3 enunciados vazios. | `conteudo/perguntasGeradas.txt` | `services/ai_client.py` | Registrar limitação ou adicionar retry |
| 2 | Dados reais dessincronizados | `[DIVERGÊNCIA]` | `perguntasGeradas.txt` (4) vs `correcao.txt` (5) vêm de execuções diferentes. | `conteudo/*.txt` | `conteudo/` | Reexecutar o fluxo E2E e preservar saídas |
| 3 | Piso heurístico em texto | `[DECISÃO DO AUTOR]` | Nota mínima 7.0 quando conceito objetivo detectado; fallback 7.5/3.0 sem LLM. | `texto_llm.py` | `evaluation/strategies/texto_llm.py` | Manter e justificar no TCC |
| 3 | Limiares 0.99/0.90 | `[DECISÃO DO AUTOR]` | Exato/aproximado para comparação de saída. | `config.py:26-27` | `correcao/config.py` | Manter e justificar |
| 3 | Pesos 70/30 na modificação | `[DECISÃO DO AUTOR]` | Execução + requisitos via LLM. | `modificacao.py` | `evaluation/strategies/modificacao.py` | Manter e justificar |
| 3 | Aprovação em `nota >= 7.0` | `[DECISÃO DO AUTOR]` | Critério de aprovação no relatório. | `formatter.py:127` | `report/formatter.py` | Documentar critério |
| Geral | Dependências reprodutíveis | `[PARCIAL]` | Sem `requirements.txt`/lock; docs citam `openai`. | `ls` raiz | raiz do projeto | Criar arquivo de dependências |
| Geral | Determinismo | `[DECISÃO DO AUTOR]` | Sem seed nos sorteios/geração. | `sorteador.py` | `gerarPergunta/utils/sorteador.py` | Manter (ou seed para estudos) |
| Geral | Roteiro experimental | `[NÃO DEFINIDO]` | Sem script/roteiro E2E gravado. | ausência | raiz | Criar roteiro se reproduzir no TCC |

---

## 12. DECISÕES QUE PRECISAM SER TOMADAS PELO AUTOR

1. **Modelo único para todas as tarefas** — uso do mesmo `qwen/qwen3-vl-4b` em geração, correção e testes; não há comparação/ablação de modelos. Decidir se isso será justificado como escolha metodológica ou se o TCC precisa de experimento com outros modelos.
2. **`MAX_TOKENS`** — enviar `max_tokens=512` na chamada da geração (para de fato materializar a configuração) ou remover a constante.
3. **Instabilidade real da geração** — a saída real teve menos perguntas que o total e enunciados vazios. Decidir se o experimento usa essa execução como está ou requer um novo pipeline com retry/validação.
4. **Piso heurístico 7.0 / fallback 7.5-3.0** — reconhecer formalmente no TCC que a nota desse fluxo ADMITE intervenção heurística quando o LLM falha.
5. **Previsão em "modo legado" (`entradaTestes`)** — o relatório real usa `modo: legado`; decidir se o TCC descreve apenas o modo pares ou os dois.
6. **Pesos 70/30 e limiares 0.99/0.90 e aprovação >= 7.0** — confirmar e justificar no texto metodológico.
7. **Desconsiderar `dados_entrada`/campos opcionais da Pergunta** — o sistema não preenche `conceito_avaliado`, `evidencia_codigo`, `resposta_esperada`, `rubrica`; decidir se isso entra como limitação explícita.
8. **Reprodutibilidade (requirements/runtime)** — criar `requirements.txt` e/ou roteiro E2E antes da banca.

---

## 13. DIVERGÊNCIAS ENTRE TCC E CÓDIGO

1. **[DIVERGÊNCIA]** `EXPLICACAO_GERAR_PERGUNTAS.md` afirma que o parser da geração reconhece apenas `"N. Título"` e que o arquivo atual (com hífen) cairia em fallback — **o código atual aceita os dois** (`knowledge_loader.py:37`: `(?:\.\s*|\s+-\s+)`). Doc desatualizado.
2. **[DIVERGÊNCIA]** `EXPLICACAO_GERAR_PERGUNTAS.md` descreve a saída com rótulo `"Seu código:"` — **o código exporta `"Código-base:"`** (`report_exporter.py:70`) e agora inclui cabeçalho de origem e, para PREVISAO, `entradaTestes:`. Doc desatualizado.
3. **[DIVERGÊNCIA]** `EXPLICACAO_GERAR_PERGUNTAS.md` indica `ARQUIVO_CONHECIMENTO = "conhecimento.txt"` a ser ajustado — **o código usa `"../conteudo/conhecimento.txt"`** (`settings.py:26`). Doc desatualizado.
4. **[DIVERGÊNCIA]** `MAX_TOKENS` (512) documentado como configurado da geração, mas **não é enviado na chamada** — reconhecido tanto na doc (`EXPLICACAO` item 5) quanto em `prompts.md` (§1), permanecendo como divergência doc↔código.
5. **[DIVERGÊNCIA]** Arquivos reais de execução dessincronizados: `conteudo/perguntasGeradas.txt` contém **4** perguntas (algumas com enunciado vazio), enquanto `conteudo/correcao.txt` corrigiu **5** exercícios — resultado de fluxos executados em momentos distintos, sem preservação de histórico.
6. **[DIVERGÊNCIA]** `prompts.md` (§3.1) documenta a variável `saida_real` vinda de `q.extras["saida_real_codigo_base"]` — **confirmado no código** (`texto_llm.py:69`; preenchida em `correcao.py:52-70`). Sem divergência.
7. **[DIVERGÊNCIA]** `prompts.md` (§3.3) afirma que PREVISÃO não usa LLM na correção — **confirmado** (determinístico; a única execução real usa `modo: legado`). Sem divergência.
8. **[CONFIRMADO/DOC]** A doc interna (`EXPLICACAO` item 4) reconhece que "o modelo não é validado semanticamente" na geração — a implementação confirma (`[NÃO DEFINIDO]` em 3.4), sendo portanto uma limitação assumida, não uma divergência não declarada.

Obs.: a documentação principal reescrita na raiz (`GERACAO_QUESTOES_TCC.md`, `CORRECAO_TCC.md`, `GERACAO_TESTES_TCC.md`) está coerente com o código verificado nesta auditoria.

---

## 14. VERIFICAÇÃO FINAL

1. **O código usa um único modelo de LLM para todas as tarefas?** Sim — `qwen/qwen3-vl-4b` (LM Studio, `localhost:1234`), na geração, na correção textual, na modificação e na geração de testes.
2. **A execução de código é isolada?** Sim — container descartável `python:3.12-alpine`, `--rm --network none --read-only --user 65534:65534`, sem montagem do projeto, com timeout 3s, CPU 3s, memória 128MB, 32 processos, 64 arquivos e limites de 64KB por fluxo; integração real confirmada pelos testes.
3. **Quantos casos de teste são alvo?** `TESTES_ALVO = 6`, com origem rastreada (`enunciado`/`llm`) e sem autocircularidade da régua.
4. **Qual a fonte da evidência da nota de PREVISÃO?** `execucao` (a nota é 100% determinística; o LLM não participa) — na execução real, similaridade 1.000.
5. **Como a modificação calcula a nota?** `nota = 0.7 × testes + 0.3 × LLM(requisitos)`; sem testes → 100% LLM; LLM falhou → parte objetiva.
6. **Quais os limiares de similaridade?** `LIMIAR_EXATO = 0.99` e `LIMIAR_APROX = 0.90`.
7. **Pergunta sem resposta do aluno é zerada?** Não — é marcada `PENDENTE` e não avaliada.
8. **O corretor textual considera "o código não tem erro" um acerto quando defensável?** Sim — o prompt instrui a verificar se o suposto erro existe, priorizar a análise técnica sobre a premissa do enunciado e considerar contestação tecnicamente correta como acerto.
9. **O `MAX_TOKENS` da geração é efetivo?** Não — declarado (512) mas não enviado na chamada; a geração real pode produzir respostas sem esse limite.
10. **Os testes automatizados passam?** Sim — 212/212 (185 em `correcao`, 27 em `gerarPergunta`) no ambiente atual, com LLM e Docker ativos.

---

*A auditoria não alterou arquivos do projeto; apenas a leu e executou as suítes de teste existentes. Este documento foi salvo por solicitação do usuário na raiz do repositório.*