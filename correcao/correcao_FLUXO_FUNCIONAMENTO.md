# Fluxo de Funcionamento - Correção

## Objetivo do fluxo

O fluxo da pasta `correcao` tem como objetivo receber questões e respostas de estudantes, aplicar a estratégia de avaliação adequada e produzir um relatório final com nota, status, feedback e detalhes da correção.

Esse módulo é híbrido: algumas respostas são avaliadas por execução de código e testes automatizados; outras são avaliadas semanticamente com apoio de um LLM.

## Visão geral do processo

```text
1. Recebimento do arquivo de entrada
2. Leitura e parsing das questões
3. Criação de objetos Questao
4. Identificação do tipo da questão
5. Escolha da estratégia de correção
6. Correção por testes, LLM ou estratégia combinada
7. Geração de nota, status e feedback
8. Criação dos objetos Resultado
9. Montagem do relatório final
10. Escrita em conteudo/correcao.txt
```

## 1. Recebimento da questão e da resposta

O processo começa com o arquivo definido em `ARQUIVO_ENTRADA`, no arquivo `config.py`.

Por padrão:

```text
../conteudo/perguntasGeradas.txt
```

Esse arquivo normalmente vem do módulo `gerarPergunta`. Depois que as perguntas são geradas, as respostas do estudante podem ser adicionadas ao mesmo arquivo.

Cada bloco pode conter:

- tipo da pergunta;
- enunciado;
- código-base;
- resposta do estudante;
- entrada esperada;
- saída esperada;
- testes explícitos.

O sistema também aceita entrada em JSON, quando as questões são fornecidas de forma estruturada.

## 2. Leitura e parsing do arquivo

A função `carregar_questoes()` lê o arquivo de entrada.

Primeiro, ela tenta interpretar o conteúdo como JSON. Se isso não funcionar, trata o arquivo como texto em blocos.

No formato textual, o parser procura estruturas como:

```text
1 - [CORRECAO] Enunciado da pergunta

Seu código:
----------------------------------------
codigo base
----------------------------------------

resposta 1 -
resposta do estudante
```

Depois da leitura, cada bloco é convertido em um objeto `Questao`.

## 3. Organização dos dados em `Questao`

Cada `Questao` reúne as informações necessárias para corrigir uma resposta.

Os campos mais importantes são:

- `idx`: número da questão;
- `tipo`: tipo declarado ou inferido;
- `enunciado`: pergunta feita ao estudante;
- `resposta_aluno`: resposta que será avaliada;
- `codigo`: código-base da questão;
- `entrada`: entrada usada em testes ou previsão;
- `saida`: saída esperada;
- `testes`: lista de casos de teste.

Essa organização permite que as estratégias de correção recebam dados padronizados.

## 4. Identificação do tipo de questão

Depois do parsing, `corrigir_questao()` identifica o tipo da questão.

A identificação segue esta ordem:

1. usa o campo `tipo`, se ele estiver claro;
2. normaliza variações do tipo com `normalizar_tipo()`;
3. se necessário, infere pelo enunciado com `inferir_tipo()`.

Tipos reconhecidos:

```text
previsao
correcao
modificacao
justificativa
descritiva
```

Se o tipo não for reconhecido, o sistema tenta um fallback: primeiro avalia como código, quando houver indícios de código, e depois usa avaliação textual via LLM.

## 5. Escolha da estratégia de correção

O arquivo `evaluation/dispatcher.py` funciona como roteador.

O mapa principal é:

| Tipo | Estratégia |
| --- | --- |
| `previsao` | `evaluation/strategies/previsao.py` |
| `correcao` | `evaluation/strategies/correcao.py` |
| `modificacao` | `evaluation/strategies/modificacao.py` |
| `justificativa` | `evaluation/strategies/justificativa.py` |
| `descritiva` | `evaluation/strategies/descritiva.py` |

A partir desse ponto, cada tipo segue um caminho próprio.

## 6. Fluxo para questão de previsão

Questões de previsão pedem que o estudante diga qual será a saída de um código.

O fluxo é:

1. localizar o código-base da questão;
2. extrair pares `Entrada/Saída` da resposta do estudante, quando existirem;
3. se necessário, extrair entradas mencionadas no enunciado;
4. executar o código-base com cada entrada;
5. remover prompts de `input()` da saída capturada;
6. comparar a saída real com a saída prevista pelo estudante;
7. calcular nota proporcional aos acertos.

Em previsão, a comparação é mais rígida, porque o objetivo é verificar se o estudante conseguiu simular mentalmente a execução do programa.

## 7. Fluxo para correção de código

Questões de correção podem pedir uma explicação textual ou uma solução em código.

O sistema primeiro verifica se o enunciado parece textual, por exemplo:

```text
Qual é o erro?
Por que o código falha?
Explique o problema.
```

Se for textual, usa avaliação via LLM.

Se for uma resposta em código, o fluxo é:

1. extrair o código da resposta do estudante;
2. obter casos de teste;
3. verificar se o código está vazio;
4. verificar sintaxe Python com `ast.parse()`;
5. executar o código em subprocesso isolado;
6. enviar entradas pelo `stdin`;
7. capturar `stdout`, `stderr`, `returncode` e timeout;
8. comparar saída obtida com saída esperada;
9. calcular nota com base na proporção de testes aprovados.

## 8. Geração e uso de testes automatizados

Os testes podem vir de duas fontes.

Primeira fonte: testes explícitos na questão, como campos `entrada`, `saida` ou `testes`.

Segunda fonte: geração via LLM, principalmente quando o código usa `input()`.

O fluxo em `tests/generator.py` é:

1. coletar testes explícitos;
2. verificar se o código usa `input()`;
3. se necessário, pedir ao LLM que gere testes;
4. validar os testes;
5. remover duplicatas;
6. limitar a quantidade a `TESTES_ALVO`.

Cada teste contém:

- `entrada`;
- `saida`;
- `obs`.

## 9. Execução segura do código

Quando uma resposta precisa ser executada, o módulo `execution/runner.py` cria um arquivo temporário com o código do estudante.

Depois, executa esse arquivo em um subprocesso com:

- interpretador Python atual;
- modo isolado (`-I`);
- entrada via `stdin`;
- captura de saída padrão e erro;
- timeout.

O retorno da execução inclui:

```text
stdout
stderr
returncode
timeout
erro_execucao
```

Esses dados são usados para decidir se o teste passou, falhou, gerou erro ou entrou em timeout.

## 10. Comparação de saídas

A estratégia de código compara a saída obtida com a saída esperada.

O sistema usa três ideias:

1. normalização de texto, removendo espaços e quebras desnecessárias;
2. cálculo de similaridade com `SequenceMatcher`;
3. verificação de presença da saída esperada dentro da saída obtida.

A terceira regra é útil quando o estudante imprime informações extras, mas ainda apresenta corretamente a saída esperada.

Exemplo:

```text
Saída esperada:
É palíndromo

Saída obtida:
Número invertido: 121
É palíndromo
```

Nesse caso, a resposta pode ser aceita em questões de código ou modificação. Em questões de previsão, a exigência é mais rígida.

## 11. Fluxo para respostas textuais

Questões `justificativa` e `descritiva` usam avaliação textual via LLM.

O fluxo é:

1. verificar se a resposta está vazia;
2. normalizar enunciado e resposta;
3. aplicar uma regra objetiva simples para reconhecer alguns conceitos centrais;
4. montar um prompt de correção textual;
5. enviar ao LLM;
6. solicitar retorno em JSON;
7. extrair nota, status, feedback, acertos e melhorias;
8. usar fallback caso o LLM não responda corretamente.

O prompt orienta o modelo a priorizar entendimento conceitual, não apenas gramática ou forma textual.

## 12. Fluxo para modificação de código

Questões de modificação combinam testes automatizados e avaliação semântica.

O fluxo é:

1. extrair o código do estudante;
2. obter ou gerar testes;
3. executar os testes;
4. calcular a nota objetiva dos testes;
5. chamar o LLM para verificar requisitos do enunciado;
6. identificar requisitos atendidos e faltantes;
7. combinar as notas.

A fórmula usada é:

```text
nota_final = 0.7 * nota_dos_testes + 0.3 * nota_do_llm
```

Essa estratégia tenta equilibrar funcionamento prático do código e atendimento conceitual ao que foi pedido.

## 13. Atribuição de nota e status

Cada estratégia retorna um objeto `Resultado`.

Esse objeto contém:

- `nota`: valor numérico de 0 a 10;
- `status`: classificação da resposta;
- `feedback`: comentário principal;
- `detalhes`: explicações complementares;
- `testes_executados`: registros dos testes, quando houver;
- `saida_correta`: saída calculada, quando aplicável.

Os status mais comuns são:

| Status | Significado |
| --- | --- |
| `ok` | Resposta correta ou plenamente satisfatória. |
| `parcial` | Resposta parcialmente correta. |
| `erro` | Resposta incorreta, vazia ou com problema relevante. |
| `falha` | Falha de avaliação, por exemplo ausência de código-base necessário. |

## 14. Geração de feedback

O feedback pode vir de diferentes fontes:

- comparação entre testes passados e falhados;
- mensagens de erro de sintaxe ou execução;
- similaridade entre saída prevista e saída real;
- análise de requisitos via LLM;
- avaliação textual via LLM;
- fallbacks heurísticos quando o LLM não responde.

O objetivo do feedback é indicar ao estudante não apenas a nota, mas também o motivo da avaliação.

## 15. Montagem do relatório final

Depois que todas as questões são corrigidas, `gerar_relatorio()` monta o arquivo final.

O relatório contém:

- título;
- data e hora da correção;
- total de questões;
- média geral;
- quantidade de questões com nota igual ou superior a 7;
- bloco individual para cada exercício.

Cada bloco individual pode conter:

- tipo da questão;
- nota;
- status;
- feedback;
- enunciado;
- saída calculada;
- detalhes;
- testes executados;
- entrada, saída esperada, saída obtida e motivo da falha ou aprovação.

## 16. Saída final

O relatório é salvo no caminho definido por `ARQUIVO_SAIDA`.

Por padrão:

```text
../conteudo/correcao.txt
```

Ao final, o terminal informa:

- caminho onde a correção foi salva;
- média geral da turma ou do conjunto de questões.

## Diagrama resumido

```text
conteudo/perguntasGeradas.txt
        |
        v
carregar_questoes()
        |
        v
Questao
        |
        v
corrigir_questao()
        |
        +--> previsao.py
        |       executa código-base e compara previsão
        |
        +--> correcao.py
        |       usa LLM textual ou testes de código
        |
        +--> modificacao.py
        |       testes 70% + LLM 30%
        |
        +--> justificativa.py / descritiva.py
                avaliação textual via LLM
        |
        v
Resultado
        |
        v
gerar_relatorio()
        |
        v
conteudo/correcao.txt
```

## Relação com o objetivo do TCC

O fluxo de correção contribui para o TCC porque mostra como LLMs podem ser integrados a técnicas tradicionais de avaliação de programação.

O sistema não abandona critérios objetivos: quando a resposta é código, ele executa e testa. Ao mesmo tempo, também avalia dimensões que testes automatizados não capturam bem, como justificativas, descrições e aderência a requisitos.

Essa combinação torna o processo mais adequado à avaliação formativa, pois gera informações úteis para aprendizagem, não apenas uma classificação final.
