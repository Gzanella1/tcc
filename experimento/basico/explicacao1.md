# Experimento básico — documentação das questões

## Visão geral

Este documento explica o experimento de nível básico do protótipo. O pipeline tem três etapas, e cada uma deixou um arquivo nesta pasta:

| Etapa | Arquivo | Conteúdo |
|---|---|---|
| Conhecimento (entrada) | `conhecimento1.txt` | 5 enunciados e o código que o aluno entregou para cada um. É aqui que os erros foram introduzidos no código. |
| Geração de perguntas | `perguntasGeradas1.txt` | Uma pergunta gerada pelo modelo para cada código (tipos PREVISAO, MODIFICACAO, DESCRITIVA, JUSTIFICATIVA e CORRECAO), o código-base usado e a resposta do aluno à pergunta gerada. |
| Correção | `correcao1.txt` | Nota, status, feedback, evidências e testes executados para cada resposta do aluno. O arquivo tem **dois blocos**, um para cada execução (veja abaixo). |

> **Como o `correcao1.txt` está organizado:**
> - **Bloco 1, de 08/11/2026 20:00:39:** tem os exercícios 1 a 4. O cabeçalho deste bloco ainda diz "Total de questões: 5", "Média geral: 5.10/10" e "Questões com nota >= 7.0: 2", mas o exercício 5 dessa execução, que tinha dado inconclusivo com nota 0.00, **não está mais no arquivo**. A média 5.10 é a de antes: (6.67 + 8.83 + 10.00 + 0.00 + 0.00) / 5 = 5.10. Ou seja, ela ainda conta o zero inconclusivo antigo da Q5.
> - **Bloco 2, de 08/11/2026 20:51:59:** é uma execução separada, só da questão 5. Ela aparece como "Total de questões: 1" e "EXERCÍCIO 1", porque a numeração recomeça dentro do bloco. Sabemos que é a Q5 porque o tipo é `correcao`, o enunciado é o da Q5 e os testes são de contagem de vogais. Resultado: nota 3.33, status erro.
>
> Nenhum dos dois blocos traz uma média que combine as cinco questões com o resultado novo da Q5. Somando 6.67 + 8.83 + 10.00 + 0.00 + 3.33 e dividindo por 5, a média ficaria em **5.77**. Esse valor é só uma conta feita aqui e **não está registrado no sistema**.
>
> **Ordem dos arquivos:** `perguntasGeradas1.txt` foi salvo às 21:07, depois de `correcao1.txt` (20:57). As saídas registradas nos testes da Q5 são as que o código atual da resposta 5 produz, então a correção é compatível com a versão atual. Mesmo assim, os arquivos não mostram qual foi a mudança feita às 21:07.

Os arquivos não dizem com que intenção cada erro foi introduzido. Os erros descritos abaixo foram identificados lendo e executando mentalmente o código de `conhecimento1.txt`, e o texto indica quando algo é uma análise e não uma informação registrada.

Para cada questão há três coisas diferentes, que não devem ser confundidas:

1. **O erro no código de entrada**, ou seja, o código original do aluno em `conhecimento1.txt`.
2. **O erro na resposta do aluno à pergunta gerada**, que está em `perguntasGeradas1.txt`.
3. **O que o sistema de correção concluiu**, registrado em `correcao1.txt`.

### Resumo

| Q | Função | Erro no código de entrada | Tipo da pergunta | Erro na resposta do aluno | Nota | Status | O sistema percebeu o problema? |
|---|---|---|---|---|---|---|---|
| 1 | `calcular_media` | Nenhum (código correto) | PREVISAO | Respondeu `7.88` em vez de `7.875` (arredondou) | 6.67 | erro | Sim: rejeitou o valor arredondado e deu nota parcial por similaridade |
| 2 | `contar_pares` | `range(1, len(lista))` ignora o índice 0 | MODIFICACAO | Manteve o erro do índice 0 | 8.83 | parcial | Não: os testes gerados têm saídas esperadas incoerentes, e o limiar de similaridade aprovou saídas erradas |
| 3 | `maior_de_tres` | Comparações estritas (`>`) falham em empates como `(5, 5, 3)` | DESCRITIVA | Afirmou que, com números iguais, a função "retorna qualquer um deles" (é falso) | 10.00 | ok | Não |
| 4 | `fatorial` | Usa `print(resultado)` em vez de `return` | JUSTIFICATIVA | Disse que "`for` só funciona pra percorrer listas e strings" | 0.00 | erro | Sim, quanto à concepção errada sobre o `for`. O `print`/`return` não foi tratado em nenhuma etapa |
| 5 | `contar_vogais` | Falta `:` no `for` (SyntaxError) | CORRECAO | Corrigiu o `:`, mas tirou as vogais maiúsculas (`"aeiou"`) | 3.33 | erro | Em parte: o teste `AEIOU` pegou a regressão, mas 3 dos 4 testes reprovados têm a saída esperada errada |

Totais: o cabeçalho do bloco 1 registra **média geral 5.10/10** e **2 questões com nota ≥ 7.0** (Q2 e Q3), mas essa média ainda conta o zero inconclusivo antigo da Q5. Com a nota nova da Q5 (3.33), a média das cinco questões seria **5.77**. Esse número foi calculado aqui e não está registrado. Continuam sendo 2 as questões com nota ≥ 7.0.

---

## Questão 1 — Média das notas (`calcular_media`)

**Erro simulado no conhecimento:**
O código de entrada **não tem erro**. Ele trata a lista vazia (`if len(notas) == 0: return 0`), soma as notas com um laço e devolve `soma / len(notas)`. Em `conhecimento1.txt`, as chamadas no final imprimem `7.875` e depois `0`, como o enunciado pede. Essa questão serve, portanto, como **caso de controle**: o código é correto, e qualquer erro vem da resposta do aluno à pergunta gerada.

**Erro ou situação na geração da pergunta:**
O modelo recebeu um código correto e gerou uma pergunta do tipo **PREVISAO**: "Qual será a saída esperada da função `calcular_media` para a entrada `[7.0, 8.5, 6.0, 10.0]`…". Como não havia defeito para apontar, a pergunta cobra o rastreamento da execução.

O **código-base** que aparece em `perguntasGeradas1.txt` não é idêntico ao de `conhecimento1.txt`. Ele mantém só a chamada `print(calcular_media([7.0, 8.5, 6.0, 10.0]))` e não tem a chamada com a lista vazia. Assim, a pergunta (que cita uma entrada só) e o código-base (que imprime uma linha só) estão de acordo.

O gerador também criou `entradaTestes: "7.0\n8.5\n6.0\n10.0\n"`. Esse campo não tem efeito: o código não lê nada da entrada padrão (não há `input()`), porque a lista está fixa na chamada `print(...)`.

**O que estamos tentando testar:**
O conceito é a **previsão da saída de um programa**: seguir o laço de soma (7.0 + 8.5 + 6.0 + 10.0 = 31.5), dividir por 4 e saber que o Python imprime o `float` resultante como `7.875`, sem arredondar. O experimento verifica duas coisas: (a) se o sistema calcula a saída correta executando o código-base, sem confiar na resposta do aluno; (b) como ele lida com uma resposta **quase certa**, um valor arredondado, em vez de simplesmente certa ou errada.

**Resposta do aluno:**
`7.88`. É o valor arredondado para duas casas, e não o que o Python imprime (`7.875`).

**Resultado da correção:**
- Nota **6.67/10**, status **erro**, feedback "Resposta incorreta para a saída esperada."
- Saída calculada a partir do código-base: `7.875`. Esse valor está correto.
- Evidência: `modo: legado`, similaridade **0.667** entre a saída do código-base e a resposta do aluno. A nota é exatamente a similaridade × 10.
- Teste 1: **FALHOU**. O relatório mostra a saída esperada (`7.875`), a resposta do aluno (`7.88`) e o motivo "saída diferente".

**Interpretação:**
O sistema calculou bem a saída de referência executando o código e não aceitou `7.88` como igual a `7.875`. Esse comportamento está de acordo com o esperado numa questão de previsão de saída, em que o valor exato importa. O relatório também deixa a comparação clara, porque mostra lado a lado a saída esperada e a resposta do aluno.

Duas observações:
- A nota parcial (6.67) mede o quanto os dois textos se parecem, e não a qualidade do raciocínio. O aluno fez a conta certa e só arredondou na hora de escrever, mas o sistema não tem como fazer essa distinção.
- O status é `erro` mesmo com uma nota de 6.67, o que mostra que o status e a nota seguem critérios diferentes. O status vem da comparação exata, e a nota vem da similaridade.

---

## Questão 2 — Contagem de pares (`contar_pares`)

**Erro simulado no conhecimento:**
O laço é `for i in range(1, len(lista))`. Ele começa no índice **1**, então o **primeiro elemento nunca é verificado**. É um erro clássico de limite (*off-by-one*). O erro fica **escondido** nos exemplos do enunciado: em `[1, 2, 3, 4, 5, 6]` e `[1, 3, 5]`, o primeiro elemento é ímpar, então ignorá-lo não muda o resultado, e o código imprime `3` e `0` corretamente. A falha só aparece quando a lista começa com um número par. Por exemplo, `[2, 4]` deveria dar 2, mas a função devolve 1.

**Erro ou situação na geração da pergunta:**
O modelo recebeu esse código e gerou uma pergunta do tipo **MODIFICACAO**: "Modifique a função para que ela também retorne uma lista contendo os próprios números pares encontrados…, mantendo a contagem de pares." A pergunta **não fala do erro do índice 0**. Ela parte do princípio de que a contagem está certa e pede uma funcionalidade nova em cima dela. O gerador não registrou nenhuma entrada de teste para essa questão.

**O que estamos tentando testar:**
O conceito é **percorrer uma lista com índices** e **acumular resultados** (um contador e uma lista com `append`), além de devolver vários valores numa tupla. Como o código-base tem um erro escondido, a questão verifica se a correção consegue **achar um defeito herdado** que o aluno não corrigiu, mesmo quando a pergunta não fala dele. Para isso, os testes gerados precisariam incluir listas que começam com número par, e o oráculo (as saídas esperadas) precisaria estar certo.

**Resposta do aluno:**
O aluno acrescentou `pares = []` e `pares.append(lista[i])`, passou a devolver `quantidade, pares` e criou um `main()` que lê os números da entrada padrão e imprime a quantidade e a lista. O laço continua `range(1, len(lista))`, então **o erro original continua lá**.

**Resultado da correção:**
- Nota **8.83/10**, status **parcial**, feedback: "A função atende ao requisito de retornar tanto a contagem quanto a lista dos pares, mantendo a ordem e os valores corretos."
- Nota composta: 70% execução de testes + 30% avaliação via LLM. 5 de 6 testes passaram, o que dá 8.33 × 0.7 = 5.83, e a nota do LLM foi 10.00 × 0.3 = 3.00. Total: 8.83.
- Os 6 testes foram gerados pelo LLM (`testes_por_origem: {'enunciado': 0, 'llm': 6}`), nenhum incompatível.
- O LLM avaliou os requisitos como `ok`, sem nada faltando.

Testes que expuseram o erro (as listas começam com número par):

| Teste | Entrada | Saída esperada (oráculo) | Correto de fato | Saída obtida | Similaridade | Resultado |
|---|---|---|---|---|---|---|
| 3 | `0 2 4 6 8` | `4` / `[0, 2, 4, 6, 8]` | `5` / `[0, 2, 4, 6, 8]` | `4` / `[2, 4, 6, 8]` | 0.903 | PASSOU |
| 4 | `10 11 12 13 14` | `2` / `[10, 12, 14]` | `3` / `[10, 12, 14]` | `2` / `[12, 14]` | 0.833 | FALHOU |
| 6 | `2 4 6 8 10 12 14` | `6` / `[2, 4, 6, 8, 10, 12, 14]` | `7` / `[2, 4, 6, 8, 10, 12, 14]` | `6` / `[4, 6, 8, 10, 12, 14]` | 0.939 | PASSOU |

Os testes 1 (`1 2 3 4 5 6`), 2 (`1 3 5`) e 5 (`7 7 7 7`) começam com número ímpar e passaram com similaridade 1.000. O erro não aparece neles.

**Interpretação:**
O sistema **não detectou** o erro herdado, e por dois motivos que se somam:

1. **O oráculo gerado pelo LLM é incoerente.** Nos testes 3, 4 e 6, a lista esperada está certa, mas a quantidade esperada está errada e coincide com o que o código com erro produz (4, 2 e 6). As observações do próprio teste contradizem os valores: o teste 4 diz "Contagem e lista de pares corretos" e espera quantidade 2 para três números pares, e o teste 6 diz "contagem exata" e espera 6 para sete números pares. Ou seja, o LLM gerou saídas esperadas que já trazem parte do erro do código.
2. **O limiar de similaridade aprovou saídas erradas.** Nos testes 3 e 6, a lista obtida não tem o primeiro elemento (`[2, 4, 6, 8]` em vez de `[0, 2, 4, 6, 8]`, e `[4, …, 14]` em vez de `[2, …, 14]`), mas o teste foi marcado como PASSOU com motivo "ok", porque a similaridade (0.903 e 0.939) ficou acima do limiar. Só o teste 4 (0.833) ficou abaixo. Os arquivos não registram o valor do limiar, mas ele fica entre 0.833 e 0.903.

Além disso, a avaliação do LLM (30% da nota) olhou só se o requisito novo foi atendido (devolver a lista). Ela deu 10.00 e um feedback que fala em "valores corretos", sem citar o erro. O resultado é que um código errado para qualquer lista que comece com número par ficou com **8.83**. Esse **não** é o comportamento esperado de um corretor. A questão mostra duas limitações: testes gerados por LLM sem verificação independente e aprovação por similaridade em vez de igualdade exata.

---

## Questão 3 — Maior de três (`maior_de_tres`)

**Erro simulado no conhecimento:**
O código só usa comparações **estritas** (`a > b and a > c`, depois `b > a and b > c`, senão `c`). Nos exemplos do enunciado, `(3, 7, 5)` dá 7 e `(10, 2, 8)` dá 10, ambos certos. Mas, analisando o código, há um caso de **empate** em que ele erra: quando `a == b` e os dois são maiores que `c`. Em `maior_de_tres(5, 5, 3)`, as duas condições são falsas (5 > 5 é falso), cai no `else` e a função devolve **3**, quando deveria devolver 5. Os outros empates (`(5, 3, 5)`, `(3, 5, 5)`, `(5, 5, 5)`) dão o resultado certo. Os arquivos não dizem se esse erro foi colocado de propósito. Ele foi identificado pela análise do código.

**Erro ou situação na geração da pergunta:**
O modelo gerou uma pergunta do tipo **DESCRITIVA**: "Descreva, com suas próprias palavras, como o código funciona passo a passo para encontrar o maior entre três números inteiros, sem usar a função max." A pergunta **não cita** empates nem casos-limite. Ela pede só uma explicação do fluxo condicional.

**O que estamos tentando testar:**
O conceito é **estruturas condicionais encadeadas** (`if`/`elif`/`else`) com operadores lógicos (`and`) e o que acontece quando as condições estritas não cobrem todos os casos. A questão verifica se a correção de uma resposta **descritiva**, feita só pelo LLM e sem testes, consegue checar se o que o aluno escreveu é verdade em relação ao código, e não apenas se o texto parece bem escrito.

**Resposta do aluno:**
A descrição do fluxo principal está certa: primeiro testa `a`, depois `b` e, se nenhum for, devolve `c`. No final, porém, o aluno escreveu: "Se tiver números iguais ele retorna qualquer um deles, porque tanto faz já que são o mesmo valor." Isso é **falso** para o código analisado, porque em `(5, 5, 3)` a função devolve `3`, que não é nenhum dos valores iguais.

**Resultado da correção:**
- Nota **10.00/10**, status **ok**, feedback "A resposta é técnica, clara e completa."
- Acerto registrado: "Descreve corretamente o fluxo lógico do código: comparação entre os três números usando condicionais."
- Melhorias: vazia. Piso aplicado: não. A única evidência é a avaliação via LLM.

**Interpretação:**
O LLM deu nota máxima e chamou a resposta de "completa" sem perceber que a afirmação sobre números iguais é falsa. Ele também não percebeu que essa afirmação aponta justamente para o defeito do código. O sistema valorizou a descrição correta do caminho principal e aceitou sem questionar uma afirmação errada sobre o caso-limite. Como questões descritivas não executam testes, não havia nada que contestasse o LLM. O resultado **não** é o esperado: a questão mostra que, em respostas abertas, a correção depende só do julgamento do LLM e pode deixar passar afirmações técnicas falsas.

---

## Questão 4 — Fatorial (`fatorial`)

**Erro simulado no conhecimento:**
O enunciado pede que a função **retorne** o fatorial, mas o código termina com `print(resultado)` e **não tem `return`**. Assim, a função devolve `None`. As chamadas `fatorial(5)` e `fatorial(0)` imprimem `120` e `1` porque o próprio `print` fica dentro da função. Por isso o erro não aparece na saída do programa, só quando alguém tenta usar o valor (por exemplo, `x = fatorial(5)` faz `x` valer `None`). O cálculo em si, com `while` e incluindo o caso `0! = 1`, está certo.

**Erro ou situação na geração da pergunta:**
O modelo gerou uma pergunta do tipo **JUSTIFICATIVA**: "Por que você escolheu usar um laço while para calcular o fatorial, em vez de um laço for ou recursão? Explique sua escolha." A pergunta é sobre a **escolha da estrutura de repetição** e **não toca** no problema do `print` em vez de `return`. Os arquivos não mostram que o gerador tenha notado esse problema.

**O que estamos tentando testar:**
Os conceitos são **estruturas de repetição** (quando usar `while`, `for` com `range`, ou recursão) e a capacidade do aluno de **justificar** uma decisão de implementação. A questão verifica se a correção de uma resposta de justificativa consegue identificar uma **concepção errada** na explicação do aluno, mesmo que o código dele funcione.

**Resposta do aluno:**
"eu usei o while pq o for so funciona pra percorrer listas e strings, com numero tem q ser while. e recursao eu nao sei fazer direito ainda". A justificativa é **tecnicamente falsa**: em Python, `for` funciona com qualquer iterável, inclusive `range(1, n + 1)`, que é a forma comum de calcular o fatorial. O aluno também diz que não domina recursão.

**Resultado da correção:**
- Nota **0.00/10**, status **erro**, feedback "Resposta incorreta e não técnica."
- Acertos: vazia. Piso aplicado: não.
- Melhorias apontadas (3):
  1. explicar que o `while` é adequado para contadores numéricos, como no fatorial, e que o `for` também pode ser usado, mas não é obrigatório;
  2. corrigir a afirmação de que "for só funciona pra percorrer listas e strings", que é falsa, porque o `for` também pode ser usado com números;
  3. explicar que a recursão é uma técnica válida e que o código pode ser reescrito de forma recursiva, mesmo que o aluno ainda não saiba fazer.

**Interpretação:**
O sistema **identificou corretamente** a concepção errada sobre o `for` e deu orientações específicas e úteis. Nesse ponto, o comportamento está de acordo com o esperado. Duas observações:
- A nota 0.00 não dá nenhum crédito parcial. O aluno escolheu uma estrutura que funciona (`while`) e foi honesto sobre não saber recursão, mas o LLM não registrou nenhum acerto. A primeira melhoria até diz que o `while` é adequado para o caso, o que reconhece que a escolha do aluno foi válida, mas isso não virou pontuação. Os arquivos não explicam o critério que levou a zero em vez de uma nota parcial.
- O erro do código de entrada (`print` em vez de `return`) **não aparece** nem na pergunta gerada nem na correção. Ele passou por todas as etapas sem ser tratado, porque a pergunta não perguntava sobre ele e a correção só avaliou a justificativa.

---

## Questão 5 — Contagem de vogais (`contar_vogais`)

**Erro simulado no conhecimento:**
A linha `for letra in texto` não tem os **dois-pontos (`:`)** no final. É um **erro de sintaxe**: o Python lança `SyntaxError` ao analisar o arquivo, e nenhuma linha chega a rodar, nem as chamadas com `"Banana"` e `"Python"`. O resto da lógica está certo e cobre maiúsculas e minúsculas (`vogais = "aeiouAEIOU"`), como pede o enunciado. Os exemplos do enunciado são `"Banana"` → 3 e `"Python"` → 1.

**Erro ou situação na geração da pergunta:**
O modelo identificou o erro e gerou uma pergunta do tipo **CORRECAO**. No entanto, o "enunciado" gerado **não é uma tarefa**. É uma afirmação que já dá a resposta: "O código apresenta um erro sintático na linha do laço for. A sintaxe correta é 'for letra in texto:', mas o código tem um erro de pontuação: falta o ':' no final da linha…". Não há pedido explícito (como "corrija o código"), e o próprio texto mostra a correção. O gerador também não criou entradas de teste nem uma saída esperada para essa questão.

**O que estamos tentando testar:**
Os conceitos são **sintaxe do Python** (os dois-pontos que abrem um bloco), percorrer uma string caractere por caractere e testar se um elemento pertence a um conjunto (`letra in vogais`), respeitando maiúsculas e minúsculas. A questão verifica duas coisas: (a) se o gerador cria uma pergunta de correção que faça o aluno encontrar o erro sozinho; (b) se a correção consegue validar o código corrigido, ou seja, se ele roda e continua cumprindo **todo** o enunciado, inclusive a parte das maiúsculas.

**Resposta do aluno:**
A resposta tem **só código**, sem nenhum texto explicando o que foi feito. Comparando com o código-base:
- o `:` foi adicionado ao `for`. Isso corrige o erro de sintaxe;
- `vogais` passou de `"aeiouAEIOU"` para `"aeiou"`. Isso é uma **regressão**, porque as maiúsculas deixam de ser contadas e o enunciado exige que sejam;
- as duas chamadas fixas (`"Banana"` e `"Python"`) foram trocadas por `texto = input()` seguido de `print(contar_vogais(texto))`. Agora o programa lê a palavra da entrada padrão.

**Resultado da correção** (bloco 2 do `correcao1.txt`, de 20:51:59, onde a questão aparece como "EXERCÍCIO 1"):
- Nota **3.33/10**, status **erro**, feedback "Apenas 2/6 testes compatíveis passaram."
- `modo: com_testes`. Foram 6 testes, todos gerados pelo LLM (`testes_por_origem: {'enunciado': 0, 'llm': 6}`), nenhum incompatível. Passaram 2 e falharam 4. A nota é 2/6 × 10 = 3.33. **Não houve avaliação via LLM** nessa questão, só a execução dos testes.

Na tabela abaixo, a coluna "Correto pelo enunciado" foi calculada aqui, contando só as letras a, e, i, o, u, maiúsculas ou minúsculas, como diz o enunciado:

| Teste | Entrada | Saída esperada (oráculo) | Observação do teste | Correto pelo enunciado | Saída obtida | Resultado | O veredito faz sentido? |
|---|---|---|---|---|---|---|---|
| 1 | `Banana` | 3 | "(a, a, a)" | 3 | 3 | PASSOU | Sim |
| 2 | `Python` | 2 | "(o, o)" | 1 (o próprio enunciado diz 1) | 1 | FALHOU | **Não**: a saída do aluno está certa e o oráculo está errado |
| 3 | `Casa` | 1 | "(a)" | 2 (a, a) | 2 | FALHOU | **Não**: a saída do aluno está certa e o oráculo está errado |
| 4 | `Programação` | 3 | "(o, a, a)" | 4 (o, a, a, o). O `ã` com acento não está em a, e, i, o, u | 4 | FALHOU | **Não**: o oráculo esquece o último `o`. Mesmo que o `ã` fosse contado, o certo seria 5, e não 3 |
| 5 | `12345` | 0 | "Sem vogais" | 0 | 0 | PASSOU | Sim |
| 6 | `AEIOU` | 5 | — | 5 | 0 | FALHOU | **Sim**: é aqui que a regressão (tirar as maiúsculas) aparece |

**Interpretação:**
Nesta execução, a resposta do aluno passou a ler a entrada com `input()`, e o sistema gerou e executou testes. Na execução anterior, a Q5 tinha ficado inconclusiva justamente por falta de "entrada que permita gerar testes". É provável que a mudança no código tenha permitido a geração de testes, mas os arquivos não dizem isso explicitamente.

- **O sistema detectou a regressão** de verdade: o teste 6 (`AEIOU` → 0, quando deveria ser 5) mostra que o aluno quebrou a contagem de maiúsculas. Esse é o comportamento esperado.
- **Mas a nota está muito mais baixa do que deveria, por erro do oráculo.** Dos 4 testes reprovados, 3 (testes 2, 3 e 4) têm uma saída esperada que **contradiz a própria contagem de vogais**. No teste 2, o oráculo contradiz até o exemplo do enunciado original (`"Python"` → 1). Nesses três casos, a saída do aluno estava certa. Se o oráculo estivesse correto, só o teste 6 teria falhado: 5/6, ou nota 8.33, em vez de 2/6 e nota 3.33.
- O sistema não comparou as saídas esperadas geradas pelo LLM com os exemplos do enunciado (nenhum teste veio do enunciado) e não percebeu que as observações dos testes não batem com as entradas (por exemplo, "(o, o)" para `Python`). É o mesmo problema da Q2, só que com o efeito contrário: na Q2, o oráculo errado **aumentou** a nota de um código com erro, e aqui ele **diminuiu** a nota de um código que estava parcialmente certo.
- O acerto principal (a correção do `:`) não foi citado no feedback. O feedback só informa quantos testes passaram.

---

## O que o experimento mostra

Com base apenas nos três arquivos:

- **O sistema executa o código-base para obter a saída de referência** (Q1). A saída calculada (`7.875`) está certa, e o relatório mostra a saída esperada ao lado da resposta do aluno. O valor arredondado (`7.88`) foi rejeitado, mas recebeu nota parcial por similaridade.
- **Erros escondidos no código de entrada não chegaram às perguntas geradas** (Q2, Q3 e Q4). Só o erro de sintaxe, que é evidente, foi apontado pelo gerador (Q5).
- **As saídas esperadas dos testes gerados pelo LLM são o principal ponto fraco da correção.** Na Q2, elas trouxeram o erro do próprio código e, junto com a **aprovação por similaridade**, deram 8.83 a um código com erro. Na Q5, três das seis saídas esperadas estavam erradas, e um código que só tinha a regressão das maiúsculas ficou com 3.33. Nos dois casos, nenhum teste foi tirado dos exemplos do enunciado, que teriam servido de checagem.
- **Correções feitas só pelo LLM** acertaram ao apontar uma concepção errada explícita (Q4), mas não perceberam uma afirmação técnica falsa num texto que, no geral, estava bem escrito (Q3).
- **Quando faltam dados para gerar testes, o sistema marca a questão como inconclusiva**, como aconteceu com a Q5 na execução das 20:00:39. Quando a resposta passou a ler a entrada, os testes foram gerados e a regressão real foi encontrada (teste `AEIOU`).
- **O relatório tem inconsistências de organização:** o cabeçalho do bloco 1 continua com a média antiga (5.10, que conta o zero inconclusivo da Q5), e a Q5 reexecutada aparece como "EXERCÍCIO 1" num bloco separado, sem uma média das cinco questões.
