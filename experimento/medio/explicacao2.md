# Explicação dos testes

Este arquivo explica **o que erramos de propósito** nos arquivos de teste e **por quê**.

- `conhecimento1.txt` → as 5 questões originais e as respostas do "aluno".
- `perguntasGeradas.txt` → as perguntas que o sistema gerou a partir dessas respostas, já com as respostas do "aluno".
- `correcao.txt` → a nota que o sistema deu para cada resposta.

## Resumo

| Questão | Erro no conhecimento | Tipo da questão gerada | Resposta na questão gerada | Sistema reconheceu o erro? | Nota | Status |
|---|---|---|---|---|---|---|
| 1 - Contagem de palavras | Nenhum (resposta certa) | Modificação | Só a 1ª letra fica minúscula | Sim, pelo LLM (os testes não pegaram) | 7.0 | parcial |
| 2 - Elementos comuns | Não ordena o resultado | Descritiva | Texto certo + uma frase falsa | Sim | 0.0 | erro |
| 3 - Segundo maior | Não trata números repetidos | Previsão | Resposta certa (`7`) | Não havia erro (deu nota máxima) | 10.0 | ok |
| 4 - Matriz transposta | Só funciona em matriz quadrada | Correção | Só código, sem explicação | Não (ignorou o código e tratou como resposta vazia) | 0.0 | erro |
| 5 - Média por aluno | `(max + min) / 2` em vez da média | Justificativa | Justificativa errada, mas convincente | Sim | 0.0 | erro |

---

## Parte 1 — Erros que colocamos no conhecimento (`conhecimento1.txt`)

### Questão 1 — Contagem de palavras
**Deixamos a resposta certa** para ver que tipo de pergunta o sistema cria quando o aluno acerta.
(Único detalhe: usa `split(' ')` em vez de `split()`, então dois espaços seguidos viram uma "palavra vazia".)

➜ O sistema gerou uma questão de **MODIFICAÇÃO** (pedir uma melhoria no código).

### Questão 2 — Elementos comuns
**Erramos: esquecemos de ordenar o resultado** (faltou o `sorted`).
O enunciado pede a lista em ordem crescente, mas o código só faz `list(conjunto)`.
O erro fica escondido, porque com números pequenos o Python costuma devolver o conjunto já em ordem.

➜ O sistema gerou uma questão **DESCRITIVA** (explicar o que o código faz).

### Questão 3 — Segundo maior
**Erramos: o código não trata números repetidos.**
- `[9, 9, 7]` devolve `9`, mas o certo é `7`.
- `[5, 5]` devolve `5`, mas o certo é `None`.

O exemplo do enunciado (`[4, 9, 2, 7]`) não tem repetição, então parece que está certo.

➜ O sistema gerou uma questão de **PREVISÃO** (dizer qual valor o código retorna).

### Questão 4 — Matriz transposta
**Erramos: o código só funciona para matriz quadrada.**
Ele usa o número de linhas para tudo e ignora o número de colunas.
- `[[1, 2, 3], [4, 5, 6]]` devolve `[[1, 4], [2, 5]]` → a terceira coluna some.
- Uma matriz 3×2 dá erro (`IndexError`).

O exemplo do enunciado é 2×2, então parece que está certo.

➜ O sistema gerou uma questão de **CORREÇÃO** (o código tem erro?).

### Questão 5 — Média por aluno
**Erramos: o código calcula `(maior + menor) / 2` em vez da média de verdade.**
No exemplo do enunciado o resultado bate por coincidência (7.0 e 7.0).
Mas com `[10, 2, 3]` a média certa é `5.0` e o código devolve `6.0`.

➜ O sistema gerou uma questão de **JUSTIFICATIVA** (explicar por que o código faz assim).

---

## Parte 2 — Respostas que colocamos nas questões geradas (`perguntasGeradas.txt`)

### Questão 1 — Modificação (deixar tudo minúsculo)
**Colocamos uma resposta meio certa:** o aluno só deixa a **primeira letra** minúscula (`"PYTHON"` vira `"pYTHON"`).
Queríamos ver se o sistema percebe uma resposta que **parece** certa, mas não está.

**Resultado: nota 7.0 (parcial).**
- Os 6 testes passaram, porque nenhum teste tinha palavra toda em maiúsculo.
- A avaliação por LLM percebeu o erro e baixou a nota.

### Questão 2 — Descritiva
**Colocamos uma explicação quase toda certa, com uma frase falsa:** que o resultado fica "na mesma ordem da lista1".
(Conjunto em Python não guarda ordem.)
Queríamos ver se o sistema acha um erro escondido no meio de um texto certo.

**Resultado: nota 0.0.** O sistema achou a frase falsa.

### Questão 3 — Previsão
**Colocamos a resposta certa (`7`).**
Queríamos ver se o sistema dá nota máxima quando o aluno acerta.

**Resultado: nota 10.0.** O sistema rodou o código, viu que dá `7` e aceitou.

### Questão 4 — Correção
**Colocamos só o código corrigido, sem nenhuma explicação.**
A pergunta pede para **explicar** o erro, não para corrigir.
Queríamos ver como o sistema reage quando o aluno responde no formato errado.

**Resultado: nota 0.0 ("Resposta vazia").** O sistema esperava texto e ignorou o código.

### Questão 5 — Justificativa
**Colocamos uma justificativa que parece convincente, mas está errada:** o aluno defende o `(maior + menor) / 2`, dizendo que é "igual à média" e "mais eficiente".
Queríamos ver se o sistema se deixa enganar por um texto bem escrito.

**Resultado: nota 0.0.** O sistema não aceitou a justificativa.
