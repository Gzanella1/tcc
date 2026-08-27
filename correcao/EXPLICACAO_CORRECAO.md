# Explicação do Sistema de Correção

Este arquivo explica como funciona o código responsável por ler as questões,
corrigir as respostas dos alunos e gerar o arquivo final `conteudo/correcao.txt`.

## Visão geral

O sistema de correção segue este fluxo:

1. Lê o arquivo de entrada definido em `ARQUIVO_ENTRADA`.
2. Transforma cada bloco do arquivo em uma `Questao`.
3. Descobre o tipo da questão.
4. Escolhe a estratégia de correção adequada.
5. Executa código, compara saídas ou chama o LLM, dependendo do tipo.
6. Gera uma lista de `Resultado`.
7. Escreve o relatório final no arquivo definido em `ARQUIVO_SAIDA`.

O ponto de entrada é:

```text
correcao/main.py
```

Por padrão, a entrada e a saída ficam configuradas em:

```text
ARQUIVO_ENTRADA = ../conteudo/perguntasGeradas.txt
ARQUIVO_SAIDA   = ../conteudo/correcao.txt
```

Esses caminhos estão em `correcao/config.py`.

## Arquivos principais

### `main.py`

É o arquivo que inicia a correção.

Ele faz três coisas principais:

1. Chama `carregar_questoes(...)` para ler as questões.
2. Chama `corrigir_questao(...)` para corrigir cada questão.
3. Chama `gerar_relatorio(...)` para criar o texto final da correção.

Se alguma questão gerar erro interno durante a correção, o sistema não para tudo.
Ele cria um `Resultado` com nota `0.0`, status `erro` e registra o erro nos detalhes.

### `config.py`

Guarda as configurações gerais do sistema.

As principais são:

```text
ARQUIVO_ENTRADA
ARQUIVO_SAIDA
LLM_BASE_URL
LLM_MODEL
LLM_TIMEOUT
USAR_LLM
LIMIAR_EXATO
LIMIAR_APROX
TESTES_ALVO
```

`LIMIAR_EXATO` é usado quando a comparação precisa ser muito rígida.
`LIMIAR_APROX` é usado quando pequenas diferenças de texto podem ser aceitas.

### `models/questao.py`

Define os dois modelos de dados mais importantes:

```python
Questao
Resultado
```

`Questao` representa uma questão carregada do arquivo de entrada.
Ela guarda informações como:

```text
idx
tipo
enunciado
resposta_aluno
codigo
entrada
saida
testes
```

`Resultado` representa a correção de uma questão.
Ele guarda:

```text
idx
tipo
nota
status
feedback
detalhes
testes_executados
saida_correta
fonte_evidencia
evidencias
```

`fonte_evidencia` indica de onde veio a evidência usada na nota:

```text
execucao   -> execução real de código ou testes
llm        -> avaliação feita pelo LLM
heuristica -> regra local, sem LLM e sem execução
ausente    -> nenhuma evidência foi usada
```

`evidencias` é uma lista de registros estruturados que explicam como a nota
foi obtida. Cada registro tem o formato:

```python
{"tipo": "...", "resumo": "...", "dados": {...}}   # "peso" opcional
```

Quando não há evidência, a lista fica vazia e a fonte é `ausente`.

## Leitura das questões

### `parsing/parser.py`

Esse arquivo transforma o conteúdo de `perguntasGeradas.txt` em objetos `Questao`.

Ele aceita dois formatos:

1. JSON.
2. Texto em blocos, como o formato gerado pelo seu sistema.

Quando o arquivo é texto, ele tenta separar os exercícios usando linhas como:

```text
========== Exercicio gerado com base na sua resposta da questão X ==========
```

Depois disso, ele procura informações importantes:

```text
[TIPO]
enunciado
código
resposta do aluno
entrada
saida
testes
```

Se o tipo não estiver claro, o sistema tenta inferir o tipo pelo texto do enunciado.

## Identificação do tipo da questão

### `utils/tipo.py`

Esse arquivo normaliza e infere o tipo da questão.

Exemplos de tipos:

```text
previsao
correcao
modificacao
justificativa
descritiva
```

Se o enunciado tiver `[MODIFICACAO]`, por exemplo, o sistema entende que a questão
é do tipo `modificacao`.

Se não tiver um marcador explícito, ele procura palavras-chave no enunciado.

## Escolha da estratégia de correção

### `evaluation/dispatcher.py`

Esse arquivo funciona como um roteador.

Ele recebe uma `Questao`, identifica o tipo e manda para a estratégia correta.

O mapa principal é:

```text
previsao      -> evaluation/strategies/previsao.py
correcao      -> evaluation/strategies/correcao.py
modificacao   -> evaluation/strategies/modificacao.py
justificativa -> evaluation/strategies/justificativa.py
descritiva    -> evaluation/strategies/descritiva.py
```

Se o tipo não for reconhecido, ele tenta corrigir como código.
Se isso não funcionar, ele usa avaliação textual com LLM.

## Correção de previsão

### `evaluation/strategies/previsao.py`

Questões de previsão pedem que o aluno diga qual será a saída de um código.

Nessas questões, o sistema:

1. Extrai a resposta do aluno.
2. Executa o código original da questão.
3. Captura a saída real.
4. Remove prompts de `input()`, quando necessário.
5. Compara a saída prevista pelo aluno com a saída real.

Aqui a comparação é mais rígida.

Isso acontece porque, em questão de previsão, o aluno precisa prever exatamente
o que o programa imprime. Se ele colocar uma saída extra que o programa não gera,
a previsão está errada.

## Correção de código

### `evaluation/strategies/codigo.py`

Esse arquivo é usado quando o aluno entrega código Python como resposta.

Ele é usado por questões de:

```text
correcao
modificacao
```

O fluxo é:

1. Verifica se o código está vazio.
2. Verifica se o código tem sintaxe Python válida.
3. Busca testes para executar.
4. Executa o código do aluno para cada teste.
5. Compara a saída obtida com a saída esperada.
6. Calcula a nota pela quantidade de testes que passaram.

Se não houver testes, ele executa o código sem entrada e aceita se não houver erro.

### Comparação de saída

A comparação principal usa `comparar_textos(...)`, de `utils/text.py`.

Ela aceita pequenas diferenças de espaço e calcula similaridade usando
`SequenceMatcher`.

Além disso, existe a função:

```python
saida_contem_esperado(saida_obtida, saida_esperada)
```

Ela foi criada para o seguinte caso:

```text
Saída esperada:
É palíndromo
Número invertido: 121

Saída obtida:
Debug: começando programa
É palíndromo
Número invertido: 121
Fim do programa
```

Nesse caso, o código do aluno imprimiu informações extras, mas a saída esperada
está presente e correta. Então o teste pode passar com o motivo:

```text
saída esperada presente; há saída extra
```

Essa tolerância vale para correção de código e modificação.
Ela não deve ser usada para previsão, porque previsão exige dizer exatamente
o que o código original imprime.

## Correção de modificação

### `evaluation/strategies/modificacao.py`

Questões de modificação pedem que o aluno altere um código para adicionar ou
mudar algum comportamento.

Essa estratégia combina duas avaliações:

```text
70% -> execução com testes
30% -> análise dos requisitos via LLM
```

Primeiro, ela usa `codigo.py` para executar o código do aluno contra casos de teste.

Depois, se `USAR_LLM` estiver ativado, ela chama o LLM para verificar se o aluno
atendeu aos requisitos do enunciado.

Exemplo:

```text
Adicione uma funcionalidade que verifique se o número é palíndromo
e exiba a versão invertida do número como saída adicional.
```

O LLM deve identificar requisitos como:

```text
verificar se é palíndromo
ignorar sinais e zeros à esquerda
exibir a versão invertida
```

A nota final é calculada assim:

```text
nota_final = 0.7 * nota_dos_testes + 0.3 * nota_do_llm
```

## Correção textual

### `evaluation/strategies/texto_llm.py`

Esse arquivo corrige respostas em texto, como justificativas ou questões
descritivas.

Ele é usado por:

```text
justificativa
descritiva
correcao textual
```

O sistema envia ao LLM:

```text
enunciado
resposta do aluno
critérios de correção
```

O LLM deve retornar JSON com:

```text
nota
status
feedback
acertos
melhorias
```

Se o LLM falhar ou não retornar JSON válido, o sistema usa uma regra de fallback.

## Geração de testes

### `tests/generator.py`

Esse arquivo monta os casos de teste usados na execução do código.

Os testes podem vir de:

1. Campos explícitos da questão, como `entrada`, `saida` ou `testes`.
2. Geração automática via LLM.

Quando o código usa `input()`, o sistema tenta gerar entradas realistas.

Cada teste possui:

```text
entrada
saida
obs
```

Internamente, cada teste também carrega a chave `_origem`, que registra a
procedência da régua de testes:

```text
enunciado -> caso explícito do enunciado ou fornecido na própria questão
llm       -> caso gerado automaticamente pelo LLM
```

Essa marcação é interna e acompanha o teste durante a validação e a
deduplicação (em uma colisão, fica com a origem do teste que sobrevive).
Na correção, a estratégia de código soma essa procedência e registra na
evidência de execução o campo:

```text
testes_por_origem: {"enunciado": X, "llm": Y}
```

Assim, o relatório permite responder quantos testes vieram do enunciado e
quantos foram gerados pelo LLM.

Antes de usar os testes, o sistema:

```text
remove testes duplicados
descarta testes sem saída esperada
descarta entrada vazia quando o código exige input()
```

## Execução do código do aluno

### `execution/runner.py`

Esse arquivo executa o código Python do aluno em um processo separado.

O sistema:

1. Cria um arquivo temporário.
2. Escreve o código do aluno nesse arquivo.
3. Executa o arquivo com o Python atual.
4. Envia a entrada do teste pelo `stdin`.
5. Captura `stdout`, `stderr`, `returncode` e timeout.

O retorno é um dicionário com:

```text
stdout
stderr
returncode
timeout
erro_execucao
```

Também existe uma função para verificar a sintaxe antes da execução:

```python
verificar_sintaxe_python(codigo)
```

## Utilidades de texto

### `utils/text.py`

Esse arquivo possui funções auxiliares usadas em várias partes da correção.

As principais são:

```text
normalizar_texto
comparar_textos
extrair_codigo
extrair_json
exige_saida_no_enunciado
extrair_prompts_input
remover_prompts_saida
saida_contem_esperado
```

`normalizar_texto(...)` remove espaços e quebras de linha desnecessárias.

`extrair_codigo(...)` pega código dentro de cercas Markdown, como:

````text
```python
print("oi")
```
````

`remover_prompts_saida(...)` remove textos vindos de `input("...")`.
Isso evita que o aluno seja penalizado porque o Python colocou o prompt no
`stdout`.

## Geração do relatório final

### `report/formatter.py`

Esse arquivo gera o conteúdo final de `conteudo/correcao.txt`.

O relatório contém:

```text
data da correção
total de questões
média geral
questões com nota >= 7
resultado de cada exercício
nota
status
feedback
enunciado
detalhes
testes executados
evidências da correção, quando houver
saída esperada
saída obtida
motivo
```

Cada questão vira um bloco separado no arquivo final.

## Resumo por tipo de questão

| Tipo | Como é corrigido |
| --- | --- |
| `previsao` | Executa o código original e compara com a saída prevista pelo aluno. |
| `correcao` | Se for código, executa testes. Se for explicação, usa LLM. |
| `modificacao` | Usa testes para o código e LLM para verificar requisitos. |
| `justificativa` | Usa avaliação textual via LLM. |
| `descritiva` | Usa avaliação textual via LLM. |

## Exemplo de fluxo completo

Imagine uma questão `[MODIFICACAO]`.

1. `main.py` lê `perguntasGeradas.txt`.
2. `parser.py` transforma o bloco em uma `Questao`.
3. `dispatcher.py` identifica o tipo `modificacao`.
4. `modificacao.py` extrai o código da resposta do aluno.
5. `tests/generator.py` obtém ou gera testes.
6. `codigo.py` executa o código do aluno.
7. `runner.py` captura a saída.
8. `utils/text.py` normaliza e compara as saídas.
9. `modificacao.py` chama o LLM para avaliar os requisitos.
10. `formatter.py` escreve o resultado no relatório final.

## Observação importante sobre prints extras

Em questões de código e modificação, um `print` extra nem sempre significa que
o código está errado.

O importante é verificar se a saída esperada aparece corretamente.

Por isso, a estratégia de código agora aceita casos em que:

```text
saida_obtida contém saida_esperada
```

desde que a execução não tenha erro, timeout ou valor incorreto.

Isso evita penalizar um aluno que imprimiu uma mensagem adicional, mas ainda
produziu a resposta pedida pelo enunciado.
