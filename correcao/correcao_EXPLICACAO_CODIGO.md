# Explicação do Código - Correção

## Visão geral

A pasta `correcao` implementa um sistema híbrido de avaliação. Ela combina regras objetivas, execução de código, testes automatizados, comparação de textos e avaliação semântica via LLM.

O fluxo principal começa em `main.py`, passa pelo parser, escolhe uma estratégia de avaliação e termina com a geração de um relatório.

```text
arquivo de entrada -> Questao -> estratégia de correção -> Resultado -> relatório
```

## `main.py`

### Função do arquivo

É o ponto de entrada da correção automática.

### Função principal

#### `main()`

Recebe:

- não recebe parâmetros diretamente;
- usa `ARQUIVO_ENTRADA` e `ARQUIVO_SAIDA`, definidos em `config.py`.

Faz:

1. chama `carregar_questoes()` para ler o arquivo de entrada;
2. percorre cada `Questao`;
3. chama `corrigir_questao()` para avaliar a resposta;
4. captura erros internos sem interromper toda a execução;
5. chama `gerar_relatorio()`;
6. salva o relatório no arquivo de saída;
7. imprime no terminal o caminho do relatório e a média geral.

Retorna:

- `0` em caso de sucesso;
- `1` se não conseguir ler a entrada ou se nenhuma questão for encontrada.

### Conexões

Conecta os principais blocos do sistema:

- `parsing/parser.py`, para entrada;
- `evaluation/dispatcher.py`, para correção;
- `report/formatter.py`, para saída.

## `config.py`

### Função do arquivo

Centraliza configurações do sistema, muitas delas controladas por variáveis de ambiente.

### Constantes principais

| Constante | Função |
| --- | --- |
| `ARQUIVO_ENTRADA` | Caminho do arquivo com questões e respostas. |
| `ARQUIVO_SAIDA` | Caminho do relatório final. |
| `LLM_BASE_URL` | URL do servidor LLM compatível com OpenAI. |
| `LLM_MODEL` | Modelo usado nas chamadas ao LLM. |
| `LLM_TIMEOUT` | Tempo máximo de espera por resposta do LLM. |
| `USAR_LLM` | Liga ou desliga o uso do LLM. |
| `LIMIAR_EXATO` | Similaridade exigida para respostas praticamente idênticas. |
| `LIMIAR_APROX` | Similaridade mínima para aceitar respostas próximas. |
| `TESTES_ALVO` | Quantidade máxima desejada de testes. |

### Conexões

É usado por quase todos os módulos: parser, estratégias, cliente LLM, executor de testes e relatório.

## `models/questao.py`

### Função do arquivo

Define os modelos de dados principais do sistema.

### Classe `Questao`

Representa uma questão carregada do arquivo de entrada.

Atributos principais:

- `idx`: identificador numérico da questão;
- `tipo`: tipo da questão;
- `enunciado`: texto da pergunta;
- `resposta_aluno`: resposta fornecida pelo estudante;
- `resposta_referencia`: gabarito textual, quando existir;
- `codigo`: código-base associado à questão;
- `entrada`: entrada de teste;
- `saida`: saída esperada;
- `testes`: lista de casos de teste;
- `extras`: campos adicionais vindos do JSON.

### Classe `Resultado`

Representa o resultado da avaliação.

Atributos principais:

- `idx`: identificador da questão corrigida;
- `tipo`: tipo da questão;
- `nota`: nota numérica;
- `status`: classificação geral, como `ok`, `parcial`, `erro` ou `falha`;
- `feedback`: comentário principal ao estudante;
- `detalhes`: observações adicionais;
- `testes_executados`: registros dos testes executados;
- `saida_correta`: saída calculada, quando aplicável;
- `fonte_evidencia`: origem da evidência usada na nota (`execucao`, `llm`, `heuristica` ou `ausente`);
- `evidencias`: lista de registros estruturados `{tipo, resumo, dados[, peso]}` que explicam como a nota foi obtida.

### Conexões

`Questao` é produzida pelo parser e consumida pelas estratégias de correção. `Resultado` é produzido pelas estratégias e consumido pelo formatador de relatório.

## `evaluation/evidencia.py`

### Função do arquivo

Centraliza a rastreabilidade da evidência. É a fonte única de verdade para os valores válidos de origem da avaliação.

### Constantes principais

| Constante | Valor | Significado |
| --- | --- | --- |
| `FONTE_EXECUCAO` | `execucao` | Nota derivada da execução real de código ou testes. |
| `FONTE_LLM` | `llm` | Nota produzida pelo LLM. |
| `FONTE_HEURISTICA` | `heuristica` | Nota por regra local, sem LLM e sem execução. |
| `FONTE_AUSENTE` | `ausente` | Nenhuma evidência foi usada (valor padrão). |

Também define os tipos de evidência válidos (`TIPO_EXECUCAO`, `TIPO_LLM`, `TIPO_HEURISTICA`) e a função:

#### `normalizar_fonte(valor)`

Recebe qualquer valor, normaliza caixa/espaços e retorna uma das fontes válidas; valores inválidos viram `ausente`.

O contrato garantido é: `evidencias == []` equivale a `fonte_evidencia == "ausente"`.

## `parsing/parser.py`

### Função do arquivo

Lê o arquivo de entrada e converte seu conteúdo em objetos `Questao`.

### Funções principais

#### `carregar_arquivo_texto(path)`

Recebe:

- `path`: caminho do arquivo.

Faz:

- verifica se o arquivo existe;
- lê o conteúdo em UTF-8.

Retorna:

- string com todo o conteúdo do arquivo.

#### `split_exercicios(texto)`

Recebe:

- texto completo do arquivo.

Faz:

- divide o conteúdo em blocos de exercícios;
- reconhece separadores em linha única ou em três linhas.

Retorna:

- lista de blocos de texto.

#### `parse_tests_field(texto)`

Recebe:

- texto representando testes.

Faz:

- tenta interpretar lista JSON;
- aceita também formatos como `entrada => saida` ou `entrada | saida`.

Retorna:

- lista de dicionários com `entrada`, `saida` e `obs`.

#### `parse_block(block, idx)`

Recebe:

- um bloco de texto;
- índice da questão.

Faz:

- identifica o cabeçalho da pergunta, como `1 - [CORRECAO] ...`;
- captura o enunciado;
- captura o bloco `Seu código`;
- captura a resposta do estudante;
- tenta extrair entradas mencionadas no enunciado.

Retorna:

- objeto `Questao`.

#### `carregar_questoes(path)`

Recebe:

- caminho do arquivo de entrada.

Faz:

1. lê o texto;
2. tenta interpretar como JSON;
3. se não conseguir, usa o parser de blocos;
4. normaliza tipos, enunciados, respostas, código e testes.

Retorna:

- lista de objetos `Questao`.

#### `extrair_entradas(enunciado)`

Recebe:

- enunciado bruto.

Faz:

- procura linhas no formato `Entrada: ...`.

Retorna:

- string com as entradas encontradas.

### Conexões

Este arquivo transforma a saída do módulo `gerarPergunta` em dados estruturados para a correção.

## `utils/tipo.py`

### Função do arquivo

Normaliza e infere o tipo da questão.

### Funções

#### `normalizar_tipo(tipo)`

Recebe:

- texto com o tipo informado.

Faz:

- remove acentos e variações;
- converte aliases para tipos canônicos.

Retorna:

- tipo padronizado, como `correcao`, `previsao` ou `modificacao`.

#### `inferir_tipo(texto)`

Recebe:

- enunciado ou texto da questão.

Faz:

- procura rótulos explícitos, como `[MODIFICACAO]`;
- procura palavras-chave relacionadas a previsão, correção, modificação, justificativa ou descrição.

Retorna:

- tipo inferido ou string vazia.

### Conexões

É usado pelo parser e pelo dispatcher para decidir a estratégia de correção.

## `utils/text.py`

### Função do arquivo

Reúne utilitários de texto usados em todo o sistema.

### Funções principais

| Função | O que faz |
| --- | --- |
| `sem_acentos()` | Remove acentos e diacríticos. |
| `normalizar_label()` | Normaliza rótulos de campos e tipos. |
| `normalizar_texto()` | Remove espaços e quebras de linha desnecessárias. |
| `compactar_texto()` | Converte texto para uma linha compacta. |
| `tokenizar()` | Separa texto em tokens alfanuméricos. |
| `comparar_textos()` | Calcula similaridade entre dois textos. |
| `saida_contem_esperado()` | Verifica se a saída obtida contém a saída esperada em ordem. |
| `extrair_codigo()` | Extrai código entre cercas Markdown. |
| `extrair_json()` | Extrai JSON de uma resposta textual. |
| `exige_saida_no_enunciado()` | Detecta se o enunciado exige saída ou retorno. |
| `codigo_tem_input()` | Verifica se um código usa `input()`. |
| `extrair_prompts_input()` | Extrai prompts literais de chamadas `input("...")`. |
| `remover_prompts_saida()` | Remove prompts de `input()` da saída capturada. |

### Conexões

Essas funções dão suporte ao parser, ao cliente LLM, à geração de testes e às estratégias de avaliação.

## `llm/client.py`

### Função do arquivo

Comunica-se com o servidor LLM por HTTP.

### Funções

#### `chamar_llm(messages, temperature, max_tokens)`

Recebe:

- lista de mensagens no formato chat;
- temperatura;
- limite de tokens.

Faz:

- monta uma requisição para `/chat/completions`;
- envia o payload ao servidor configurado;
- lê a resposta.

Retorna:

- texto gerado pelo LLM ou `None` em caso de falha.

#### `chamar_llm_json(messages, temperature, max_tokens)`

Recebe:

- os mesmos dados de `chamar_llm()`.

Faz:

- chama o LLM;
- tenta extrair JSON da resposta.

Retorna:

- objeto Python parseado ou `None`.

### Conexões

É usado por `texto_llm.py`, `modificacao.py` e `tests/generator.py`.

## `execution/runner.py`

### Função do arquivo

Executa código Python do estudante de forma isolada.

### Funções

#### `verificar_sintaxe_python(codigo)`

Recebe:

- código Python como string.

Faz:

- usa `ast.parse()` para verificar sintaxe.

Retorna:

- tupla `(True, "")` se a sintaxe for válida;
- tupla `(False, mensagem)` se houver erro.

#### `executar_codigo_python(codigo, entrada, timeout)`

Recebe:

- código Python;
- entrada enviada ao `stdin`;
- limite de tempo.

Faz:

- cria um diretório temporário;
- grava o código em arquivo;
- executa o arquivo com o interpretador Python atual em modo isolado;
- captura `stdout`, `stderr`, `returncode`, timeout e erro.

Retorna:

- dicionário com dados da execução.

#### `executar_codigo_python_sem_entrada(codigo, timeout)`

Recebe:

- código Python;
- timeout.

Faz:

- chama `executar_codigo_python()` com entrada vazia.

Retorna:

- dicionário de execução.

### Conexões

É usado pelas estratégias de código e previsão.

## `tests/generator.py`

### Função do arquivo

Obtém casos de teste para avaliar código.

### Funções principais

#### `deduplicar_testes(testes)`

Remove testes com entradas repetidas. A chave interna `_origem` acompanha o teste que sobrevive à remoção.

#### `validar_testes(testes, requer_input)`

Filtra testes inválidos, descartando casos sem saída esperada ou sem entrada quando o código exige `input()`. Preserva a procedência `_origem` do teste original.

Cada teste carrega a chave interna `_origem` (`ORIGEM_ENUNCIADO = "enunciado"` ou `ORIGEM_LLM = "llm"`), atribuída no ponto em que o teste é criado: casos explícitos do enunciado e testes da própria `Questao` recebem `enunciado`; casos gerados via LLM recebem `llm`. Essa marcação não faz parte da interface pública (`entrada`/`saida`/`obs`) e nunca aparece nos testes executados nem no relatório; ela alimenta a contagem `testes_por_origem` na evidência de execução.

#### `_gerar_testes_llm_once(q, quantidade)`

Chama o LLM uma vez para gerar testes.

Recebe:

- objeto `Questao`;
- quantidade desejada de testes.

Retorna:

- lista de testes válidos.

#### `gerar_testes_com_llm(q, quantidade)`

Tenta gerar testes com LLM em até três tentativas.

#### `obter_testes_explicitos(q)`

Busca testes já declarados na questão, nos campos `entrada`, `saida` ou `testes`.

#### `obter_testes(q)`

Combina testes explícitos e testes gerados pelo LLM, valida, deduplica e limita a `TESTES_ALVO`.

### Conexões

É usado por `correcao.py`, `modificacao.py` e pelo fallback do dispatcher.

## `evaluation/dispatcher.py`

### Função do arquivo

Seleciona a estratégia de correção adequada para cada questão.

### Funções

#### `_extrair_resposta_codigo(q)`

Recebe:

- objeto `Questao`.

Faz:

- limpa rótulos como `Código:`;
- extrai código de cercas Markdown;
- retorna o código encontrado.

#### `corrigir_questao(q)`

Recebe:

- objeto `Questao`.

Faz:

1. normaliza ou infere o tipo;
2. procura a estratégia correspondente;
3. chama o avaliador correto;
4. se o tipo for desconhecido, tenta avaliar como código;
5. se isso não funcionar, usa avaliação textual via LLM.

Retorna:

- objeto `Resultado`.

### Conexões

É chamado por `main.py` e chama os módulos em `evaluation/strategies/`.

## `evaluation/strategies/codigo.py`

### Função do arquivo

Avalia respostas em código Python por execução contra casos de teste.

### Funções

#### `_detectar_erro_execucao(execucao)`

Recebe:

- dicionário retornado pelo executor.

Faz:

- verifica timeout, erro explícito, `returncode` diferente de zero e mensagens em `stderr`.

Retorna:

- tupla `(houve_erro, motivo)`.

#### `avaliar(q, codigo_aluno, testes)`

Recebe:

- objeto `Questao`;
- código do estudante;
- lista de testes.

Faz:

1. rejeita código vazio;
2. verifica sintaxe;
3. se não houver testes, executa o código sem entrada;
4. se houver testes, executa cada caso;
5. remove prompts de `input()` antes de comparar;
6. compara saída obtida e esperada;
7. calcula nota proporcional aos testes aprovados.

Além do resultado da correção, `avaliar()` registra evidências estruturadas em `Resultado.evidencias`: erro de sintaxe, execução sem testes ou contagem de casos no modo com testes. Nesse último modo, os `dados` incluem `testes_por_origem` (`{"enunciado": X, "llm": Y}`), somando a procedência interna `_origem` de cada teste; casos sem a chave são contados como `enunciado`. A soma X + Y é sempre igual ao total de testes executados.

Retorna:

- objeto `Resultado`.

### Conexões

É usado por `correcao.py`, `modificacao.py` e pelo fallback do dispatcher.

## `evaluation/strategies/previsao.py`

### Função do arquivo

Corrige questões em que o estudante deve prever a saída de um código.

### Funções importantes

| Função | O que faz |
| --- | --- |
| `_extrair_pares_resposta()` | Extrai pares `Entrada/Saída` da resposta do estudante. |
| `_extrair_entradas_enunciado()` | Procura entradas mencionadas no enunciado. |
| `_extrair_blocos_saida_resposta()` | Divide a resposta em blocos de saída. |
| `_montar_pares_enunciado_resposta()` | Combina entradas do enunciado com saídas respondidas. |
| `_extrair_prompts_input()` | Extrai prompts de `input()` no código-base. |
| `_remover_prompts_saida()` | Remove prompts antes da comparação. |
| `_avaliar_com_pares()` | Executa o código para cada entrada e compara a previsão. |
| `_avaliar_modo_legado()` | Fallback quando não há pares extraíveis. |
| `avaliar()` | Ponto de entrada da estratégia. |

### Dados recebidos e retornados

`avaliar(q)` recebe uma `Questao` e retorna um `Resultado`.

A comparação em previsão é mais rígida, porque o estudante deve informar a saída real do programa, não apenas uma resposta parecida.

## `evaluation/strategies/correcao.py`

### Função do arquivo

Corrige questões do tipo `correcao`.

### Funções

#### `_pergunta_eh_textual(enunciado)`

Detecta se a questão pede explicação textual, como "qual é o erro?" ou "por que falha?".

#### `_extrair_codigo_resposta(resposta_aluno)`

Extrai código da resposta do estudante.

#### `avaliar(q)`

Recebe:

- objeto `Questao`.

Faz:

- se a pergunta for textual, chama `texto_llm.avaliar()`;
- caso contrário, obtém testes e extrai o código do estudante;
- se não houver testes, usa avaliação textual;
- se houver testes, chama a estratégia de código.

Retorna:

- objeto `Resultado`.

## `evaluation/strategies/modificacao.py`

### Função do arquivo

Corrige questões em que o estudante modifica um código.

### Funções

#### `_extrair_codigo_resposta(resposta_aluno)`

Extrai o código da resposta.

#### `_avaliar_requisitos_llm(q, codigo_aluno)`

Recebe:

- objeto `Questao`;
- código do estudante.

Faz:

- monta prompt para o LLM identificar requisitos atendidos e faltantes;
- pede retorno em JSON.

Retorna:

- dicionário com nota, status, requisitos e feedback, ou `None`.

#### `avaliar(q)`

Faz:

1. obtém testes;
2. extrai o código do estudante;
3. avalia o código por testes;
4. se o LLM estiver ativo, avalia requisitos;
5. calcula a nota final.

Retorna:

- objeto `Resultado`.

### Regra de nota

```text
nota_final = 0.7 * nota_dos_testes + 0.3 * nota_do_llm
```

## `evaluation/strategies/texto_llm.py`

### Função do arquivo

Avalia respostas textuais com apoio do LLM.

### Funções

#### `_detectar_conceito_ok(enun_low, resp_low)`

Aplica uma regra objetiva simples para reconhecer alguns conceitos centrais, especialmente em questões sobre maiúsculas, minúsculas, vogais e padronização.

#### `avaliar(q)`

Recebe:

- objeto `Questao`.

Faz:

1. valida se a resposta não está vazia;
2. normaliza enunciado e resposta;
3. aplica regra objetiva de conceito;
4. monta prompt de correção textual;
5. chama o LLM;
6. interpreta JSON com nota, status, feedback, acertos e melhorias;
7. usa fallback se o LLM falhar.

Retorna:

- objeto `Resultado`.

## `evaluation/strategies/descritiva.py`

### Função do arquivo

Define a estratégia de questões descritivas.

### Funcionamento

Esse arquivo reexporta diretamente a função `avaliar` de `texto_llm.py`.

Recebe:

- objeto `Questao`.

Retorna:

- objeto `Resultado`.

## `evaluation/strategies/justificativa.py`

### Função do arquivo

Define a estratégia de questões de justificativa.

### Funcionamento

Assim como `descritiva.py`, reaproveita a avaliação textual implementada em `texto_llm.py`.

## `evaluation/strategies/dispatcher.py`

### Função do arquivo

Este arquivo possui conteúdo equivalente ao `evaluation/dispatcher.py`. No fluxo atual, `main.py` importa `evaluation.dispatcher`, portanto o dispatcher fora da pasta `strategies` é o roteador efetivamente usado.

Sua presença indica duplicação estrutural que pode ser revisada futuramente, mas não altera o funcionamento descrito.

## `evaluation/correctors.py`

### Função do arquivo

Reúne avaliadores em um formato mais concentrado, como:

- `avaliar_previsao`;
- `avaliar_texto_heuristico`;
- `avaliar_texto_llm`;
- `avaliar_codigo_por_testes`;
- `avaliar_modificacao_com_llm`.

No desenho atual, o fluxo principal usa os módulos de `evaluation/strategies/`. Por isso, `correctors.py` pode ser entendido como uma versão anterior, referência ou apoio para a lógica de correção.

## `report/formatter.py`

### Função do arquivo

Formata os resultados em um relatório textual.

### Funções

#### `formatar_resultado(res, q)`

Recebe:

- objeto `Resultado`;
- objeto `Questao`.

Faz:

- monta o bloco textual de uma questão;
- inclui nota, status, feedback, enunciado, detalhes, saída calculada e testes executados;
- quando `Resultado.evidencias` não estiver vazio, adiciona a seção "Evidências:", com uma linha por evidência no formato `- [tipo | peso] resumo`, seguida dos campos de `dados` (valores longos são truncados).

Retorna:

- string formatada.

#### `gerar_relatorio(questoes, resultados)`

Recebe:

- lista de questões;
- lista de resultados.

Faz:

- calcula média geral;
- conta quantas questões tiveram nota maior ou igual a 7;
- monta o cabeçalho do relatório;
- adiciona o bloco de cada questão.

Retorna:

- string com o relatório completo.

## Partes mais importantes para entender o sistema

1. `parser.py` transforma arquivos em objetos `Questao`.
2. `dispatcher.py` decide qual estratégia será usada.
3. `codigo.py` executa código e compara saídas.
4. `previsao.py` calcula a saída real do código-base e compara com a previsão do estudante.
5. `texto_llm.py` avalia respostas abertas com apoio semântico.
6. `modificacao.py` combina testes automatizados e análise de requisitos.
7. `evidencia.py` garante que cada nota indique de onde veio sua evidência, com registros estruturados e rastreáveis.
8. `formatter.py` transforma os resultados em relatório compreensível.

Esses arquivos mostram a principal contribuição técnica da pasta `correcao`: integrar avaliação objetiva e avaliação semântica em um mesmo fluxo de correção formativa.
