# Registro de Questões, Respostas e Avaliações


1. [Questões-base — Lista de Algoritmos do Prof. Rodack](#1-questões-base--lista-de-algoritmos-do-prof-rodack)
2. [Perguntas Geradas](#2-perguntas-geradas)
3. [Correção Automática](#3-correção-automática)

---

# 1. Questões-base: retirada da lista de Algoritimo 



## Questão 1 — Caracteres únicos

**Enunciado:**

> Escreva uma função Python para verificar se uma string possui caracteres únicos. Por exemplo, a string `"azul"` não repete letras, mas a string `"ferramenta"` possui letras repetidas. A função deve receber uma string e retornar `True` no primeiro caso (letras únicas) ou `False` caso contrário (letras repetidas). Você deve usar conjuntos para implementar a função.

```python
def verifica_palavra(s):
    conjuntos= set()
    for carater in s:
        conjuntos.add(carater)
    if len(conjuntos)== len(s):
        return True
    return False

def main():
    s=input("digite uma palavra: ")
    print(verifica_palavra(s))

main()

```

## Questão 2 — Diferença simétrica

**Enunciado:**

> Escreva uma função Python que recebe dois conjuntos de números inteiros `M` e `N` e retorna uma lista com sua diferença simétrica em ordem ascendente. O termo diferença simétrica denota os valores que existem nos conjuntos `M` ou `N`, mas não existem em ambos os conjuntos simultaneamente. Por exemplo, para os conjuntos `M = {2, 4, 5, 9}` e `N = {2, 4, 11, 12}`, a resposta deveria ser `[5, 9, 11, 12]`.

```python
def diferenca_simetrica(m,n):
    c1 = m.union(n)
    c2 = m.intersection(n)

    return c1.difference(c2)

def main():
    m={1,2,3,4,5,6}
    n={1,2,7,8,9,10}
    print(diferenca_simetrica(m,n))
main()

```

## Questão 3 — Busca reversa

**Enunciado:**

> Escreva uma função chamada `buscaReversa`, que encontra todas as chaves de um dicionário que estão mapeadas para um determinado valor. A função deve receber como parâmetros um dicionário e um valor para ser buscado no dicionário. A função deve retornar uma lista (possivelmente vazia) com as chaves encontradas. Escreva uma função `main` para demonstrar sua função. Note que a função deve funcionar independentemente de ela retornar múltiplas chaves, uma única chave, ou nenhuma chave.

```python
def busca_por_valores(d1,n):
    lista=[]
    for x, y in d1.items():
        if y==n:
            lista.append(x)

    return(lista)


def main():
    d1={}
    while True:
        chave=input("digite uma chave: ")
        if chave =="":
            break
        valor=(input("digite um valor: "))
        d1[chave]=valor
    n=input("digite o valor para buscar as chaves: ")
    chaves=busca_por_valores(d1,n)
    print(chaves)

main()

```

## Questão 4 — Ordem crescente

**Enunciado:**

> Escreva um programa Python que leia números inteiros (do usuário) e os armazena em uma lista. Seu programa deve continuar lendo inteiros até que o usuário entre com o valor `0` (zero). Então, o programa deve exibir em ordem crescente todos os números digitados pelo usuário sem incluir o valor `0`, com um valor em cada linha impressa.
>
> **Observação:** você pode implementar o algoritmo de ordenação BubbleSort ou usar o método `sort` ou a função `sorted` para ordenar a lista.

```python
lista = []
while True:
    n = int(input("digite um numero:"))
    if (n==0):
        break
    else:
        lista.append(n)

novalista =sorted(lista)
for i in novalista:
    print(i)


```

## Questão 5 — Ordem decrescente

**Enunciado:**

> Escreva um programa Python que leia números inteiros (do usuário) e os armazena em uma lista. Seu programa deve continuar lendo inteiros até que o usuário entre com o valor `0` (zero). Então, o programa deve exibir em ordem decrescente todos os números digitados pelo usuário sem incluir o valor `0`, com um valor em cada linha impressa.

```python

lista = []
while True:
    n = int(input("digite um numero:"))
    if (n == 0):
        break
    else:
        lista.append(n)
lista.sort(reverse=True)
for i in lista:
    print(i)
```

# 2. Perguntas Geradas


## Pergunta 1 — JUSTIFICATIVA

**Gerada com base na resposta da Questão 1**

### Enunciado

> **Por que você escolheu usar um conjunto (`set`) para verificar se todos os caracteres na string são únicos? Explique a vantagem dessa abordagem em relação a outras possibilidades.**

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

### Resposta do aluno

> Eu escolhi usar um conjunto (set) porque ele não permite elementos repetidos. tenho um conjunto e Se forem iguais, significa que todos os caracteres são únicos.

---

## Pergunta 2 — PREVISAO

**Gerada com base na resposta da Questão 2**

### Enunciado

> **Dados os conjuntos `m = {1, 2, 3, 4, 5, 6}` e `n = {1, 2, 7, 8, 9, 10}`, qual será a saída esperada da função `diferenca_simetrica(m, n)`?**

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

### Entrada de testes

```text
1
2
3
4
5
6
1
2
7
8
9
10
```

### Resposta do aluno

```text
{3, 4, 5, 6, 7, 8, 9, 10}
```

---

## Pergunta 3 — MODIFICACAO

**Gerada com base na resposta da Questão 3**

### Enunciado

> **Modifique a função `buscaReversa` para que, ao invés de buscar chaves por valor, ela busque valores por chave, ou seja, dado um dicionário e uma chave, retorne o valor associado a essa chave. Como faria essa alteração?**

### Código-base

```python
def busca_por_valores(d1, n):
    lista = []

    for x, y in d1.items():
        if y == n:
            lista.append(x)

    return lista


def main():
    d1 = {}

    while True:
        chave = input("digite uma chave: ")

        if chave == "":
            break

        valor = input("digite um valor: ")
        d1[chave] = valor

    n = input("digite o valor para buscar as chaves: ")
    chaves = busca_por_valores(d1, n)

    print(chaves)


main()
```

### Resposta do aluno

```python
def busca_reversa(d1, chave):
    return d1[chave]


def main():
    d1 = {}

    while True:
        chave = input("digite uma chave: ")

        if chave == "":
            break

        valor = input("digite um valor: ")
        d1[chave] = valor

    chave = input("digite a chave para buscar o valor: ")
    valor = busca_reversa(d1, chave)

    print(valor)


main()
```

---

## Pergunta 4 — CORRECAO

**Gerada com base na resposta da Questão 4**

### Enunciado

> **O código apresenta um erro real? Se sim, qual é a natureza do erro e como ele afeta o comportamento esperado do programa?**

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

### Resposta do aluno

> Não, o código não apresenta um erro de lógica.
>
> Ele funciona corretamente quando o usuário digita números inteiros.
>
> O programa para quando é digitado 0, ordena os números da lista e depois imprime os valores em ordem crescente.

---

## Pergunta 5 — DESCRITIVA

**Gerada com base na resposta da Questão 5**

### Enunciado

> **Descreva, com suas próprias palavras, o que o código faz passo a passo, explicando cada parte do programa.**

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

### Resposta do aluno

> O código começa criando uma lista vazia. Depois, ele pede números ao usuário dentro de um laço while. Quando o usuário digita 0, o programa para de pedir números. Os outros números são adicionados na lista. Depois, a lista é organizada em ordem decrescente com `sort(reverse=True)`.
>
> Por fim, o `for` percorre a lista e mostra cada número na tela.

---

# 3. CORREÇÃO AUTOMÁTICA

**Data:** 07/09/2026 19:55:05  
**Total de questões:** 5  
**Média geral:** 7.77/10  
**Questões com nota >= 7.0:** 4

---

## EXERCÍCIO 1

**Tipo:** `justificativa`  
**Nota:** 0.00/10  
**Status:** `erro`  
**Feedback:** Resposta incompleta e incorreta no conceito técnico.

### Enunciado

> Por que você escolheu usar um conjunto (set) para verificar  
> se todos os caracteres na string são únicos? Explique a vantagem dessa abordagem  
> em relação a outras possibilidades.

### Detalhes

- **Melhorias:**
  - Explique por que o uso de `set` é eficiente para verificar unicidade de caracteres (complexidade O(n)).
  - Mencione que o tamanho do `set` é comparado ao tamanho da string para verificar se há repetições.
  - Corrija a lógica: `'Se forem iguais, significa que todos os caracteres são únicos'` — isso é verdade, mas precisa ser explicado com precisão.

### Evidências

- **[llm]** Avaliação via LLM (nota original 0.00).
  - `nota_original`: `0.0`
  - `status`: `erro`
  - `acertos`: *(vazia)*
  - `melhorias`: Explique por que o uso de `set` é eficiente para verificar unicidade de caracteres (complexidade O(n).); Mencione que o tamanho do set é comparado ao tamanho da string para verificar se há repetições.; Corrija a lógica: 'Se forem iguais, significa que todos os caracteres são únicos' — isso é verdade, mas precisa ser explicado com precisão.
  - `piso_aplicado`: `não`

---

## EXERCÍCIO 2

**Tipo:** `previsao`  
**Nota:** 10.00/10  
**Status:** `ok`  
**Feedback:** Resposta correta.

### Enunciado

> Dado os conjuntos `m = {1, 2, 3, 4, 5, 6}` e `n = {1, 2, 7, 8, 9, 10}`, qual será a saída esperada da função `diferenca_simetrica(m,n)`?

### Saída calculada

```text
{3, 4, 5, 6, 7, 8, 9, 10}
```

### Detalhes

- Saída correta calculada a partir do código-base.
- Similaridade com a resposta do aluno: `1.000`

### Evidências

- **[execucao]** Saída do código-base comparada diretamente com a resposta do aluno (similaridade=1.000).
  - `modo`: `legado`
  - `similaridade`: `1.0`
  - `referencia`: `testes_executados`

### Testes executados

#### Teste 1 — PASSOU

**Entrada:**

```text
1
2
3
4
5
6
1
2
7
8
9
10
```

**Saída esperada:**

```text
{3, 4, 5, 6, 7, 8, 9, 10}
```

**Saída obtida:**

```text
{3, 4, 5, 6, 7, 8, 9, 10}
```

**Motivo:** `ok`

---

## EXERCÍCIO 3

**Tipo:** `modificacao`  
**Nota:** 8.83/10  
**Status:** `parcial`  
**Feedback:** A função atende exatamente ao requisito: dado um dicionário e uma chave, retorna o valor associado à chave.

### Enunciado

> Modifique a função `buscaReversa` para que, ao invés de buscar chaves por valor, ela busque valores por chave, ou seja, dado um dicionário e uma chave, retorne o valor associado à chave. Como faria essa alteração?

### Detalhes

- **Teste 1:** PASSOU (similaridade=0.250, motivo=saída esperada presente; há saída extra)
- **Teste 2:** FALHOU (similaridade=0.194, motivo=Processo retornou código diferente de zero)
- **Teste 3:** PASSOU (similaridade=0.250, motivo=saída esperada presente; há saída extra)
- **Teste 4:** PASSOU (similaridade=0.250, motivo=saída esperada presente; há saída extra)
- **Teste 5:** PASSOU (similaridade=0.250, motivo=saída esperada presente; há saída extra)
- **Teste 6:** PASSOU (similaridade=0.250, motivo=saída esperada presente; há saída extra)
- **Requisitos atendidos:** A função `busca_reversa` recebe um dicionário e uma chave, e retorna o valor associado àquela chave.
- **Nota composta:** 70% execução de testes + 30% avaliação via LLM.

### Evidências

- **[execucao | peso 0.7]** 5/6 casos de teste compatíveis passaram de 6 válidos (6 total, 0 incompatíveis).
  - `modo`: `com_testes`
  - `testes_total`: `6`
  - `testes_validos`: `6`
  - `testes_passaram`: `5`
  - `testes_reprovados`: `1`
  - `testes_incompativeis`: `0`
  - `testes_por_origem`: `{'enunciado': 0, 'llm': 6}`
  - `referencia`: `testes_executados`

- **[llm | peso 0.3]** Verificação dos requisitos do enunciado via LLM (nota 10.00, status `ok`).
  - `nota`: `10.0`
  - `status`: `ok`
  - `requisitos_identificados`: Modifique a função buscaReversa para que, ao invés de buscar chaves por valor, ela busque valores por chave
  - `requisitos_atendidos`: A função busca_reversa recebe um dicionário e uma chave, e retorna o valor associado àquela chave.
  - `faltantes`: *(vazia)*

### Testes executados

#### Teste 1 — PASSOU

**Obs:** Busca valor associado à chave `'chave1'` em um dicionário com duas entradas.

**Entrada:**

```text
chave1
valor1
chave2
valor2

chave1
```

**Saída esperada:**

```text
valor1
```

**Saída obtida:**

```text
digite a chave para buscar o valor: valor1
```

**Motivo:** saída esperada presente; há saída extra

---

#### Teste 2 — FALHOU

**Obs:** Chave não existe no dicionário, deve gerar `KeyError`.

**Entrada:**

```text
chaveA
valorA
chaveB
valorB

chaveC
```

**Saída esperada:**

```text
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'chaveC'
```

**Saída obtida:**

```text
digite a chave para buscar o valor:
```

**Motivo:** Processo retornou código diferente de zero

**STDERR:**

```text
Traceback (most recent call last):
  File "/work/resposta_aluno.py", line 18, in <module>
    main()
  File "/work/resposta_aluno.py", line 15, in main
    valor=busca_reversa(d1,chave)
          ^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/resposta_aluno.py", line 2, in busca_reversa
    return d1[chave]
           ~~^^^^^^^
KeyError: 'chaveC'
```

---

#### Teste 3 — PASSOU

**Obs:** Busca valor associado à chave que já existe no dicionário.

**Entrada:**

```text
chaveX
valorX
chaveY
valorY

chaveX
```

**Saída esperada:**

```text
valorX
```

**Saída obtida:**

```text
digite a chave para buscar o valor: valorX
```

**Motivo:** saída esperada presente; há saída extra

---

#### Teste 4 — PASSOU

**Obs:** Busca valor associado à chave que está no meio do dicionário.

**Entrada:**

```text
chave1
valor1
chave2
valor2
chave3
valor3

chave2
```

**Saída esperada:**

```text
valor2
```

**Saída obtida:**

```text
digite a chave para buscar o valor: valor2
```

**Motivo:** saída esperada presente; há saída extra

---

#### Teste 5 — PASSOU

**Obs:** Busca valor associado à chave em posição central do dicionário.

**Entrada:**

```text
chaveA
valorA
chaveB
valorB
chaveC
valorC

chaveB
```

**Saída esperada:**

```text
valorB
```

**Saída obtida:**

```text
digite a chave para buscar o valor: valorB
```

**Motivo:** saída esperada presente; há saída extra

---

#### Teste 6 — PASSOU

**Obs:** Busca valor associado à única chave inserida no dicionário.

**Entrada:**

```text
chaveZ
valorZ

chaveZ
```

**Saída esperada:**

```text
valorZ
```

**Saída obtida:**

```text
digite a chave para buscar o valor: valorZ
```

**Motivo:** saída esperada presente; há saída extra

---

## EXERCÍCIO 4

**Tipo:** `correcao`  
**Nota:** 10.00/10  
**Status:** `ok`  
**Feedback:** Resposta correta e técnica.

### Enunciado

> O código apresenta um erro real? Se sim, qual é a natureza do erro e como ele afeta o comportamento esperado do programa?

### Detalhes

- **Acertos:**
  - Identificou corretamente que o código não apresenta erro de lógica.
  - Explicou com precisão o comportamento do programa: parada em 0, ordenação e impressão.
  - Não assumiu premissas do enunciado sem análise técnica.
  - Resposta técnica e objetiva, sem ambiguidade.

### Evidências

- **[llm]** Avaliação via LLM (nota original 100.00).
  - `nota_original`: `100.0`
  - `status`: `ok`
  - `acertos`: Identificou corretamente que o código não apresenta erro de lógica.; Explicou com precisão o comportamento do programa: parada em 0, ordenação e impressão.; Não assumiu premissas do enunciado sem análise técnica. …(+1)
  - `melhorias`: *(vazia)*
  - `piso_aplicado`: `não`

---

## EXERCÍCIO 5

**Tipo:** `descritiva`  
**Nota:** 10.00/10  
**Status:** `ok`  
**Feedback:** A resposta é técnica, clara e completa.

### Enunciado

> Descreva, com suas próprias palavras, o que o código faz passo a passo, explicando cada parte do programa.

### Detalhes

- **Acertos:**
  - Descreve corretamente o funcionamento do código passo a passo.
  - Identifica a criação da lista vazia.
  - Explica o laço `while` e o uso do `break` com o valor 0.
  - Menciona a ordenação decrescente com `sort(reverse=True)`.
  - Aponta que o `for` percorre e exibe os elementos da lista.

### Evidências

- **[llm]** Avaliação via LLM (nota original 10.00).
  - `nota_original`: `10.0`
  - `status`: `ok`
  - `acertos`: Descreve corretamente o funcionamento do código passo a passo.; Identifica a criação da lista vazia.; Explica o laço `while` e o uso do `break` com o valor 0. …(+2)
  - `melhorias`: *(vazia)*
  - `piso_aplicado`: `não`

---
