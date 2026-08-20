# Explicacao do Gerador de Perguntas

Este arquivo explica como funciona o modulo `gerarPergunta`, como as perguntas sao geradas, quais arquivos participam do processo e quais pontos merecem atencao no estado atual do projeto.

## Ideia geral

O gerador de perguntas transforma respostas/codigos de alunos em perguntas pedagogicas feitas por IA.

Ele nao cria perguntas sozinho apenas com regras fixas. O sistema faz uma combinacao de:

- exercicios e codigos lidos de um arquivo de conhecimento;
- tipos de pergunta sorteados pelo sistema;
- prompts montados de forma especifica para cada tipo;
- resposta de um modelo de linguagem rodando pelo LM Studio;
- exportacao final para um arquivo `.txt`.

Fluxo resumido:

```text
arquivo de conhecimento
        |
        v
KnowledgeLoader
        |
        v
lista de Exercicio
        |
        v
QuestionOrchestrator
        |
        +--> Sorteador escolhe tipos de pergunta
        |
        +--> QuestionBuilder monta prompts
        |
        +--> AIClient chama a IA
        |
        v
lista de Pergunta
        |
        v
ReportExporter
        |
        v
conteudo/perguntasGeradas.txt
```

## Arquivos principais

| Arquivo | Funcao |
|---|---|
| `gerarPergunta/main.py` | Ponto de entrada. Liga as partes: carrega exercicios, gera perguntas e salva o relatorio. |
| `gerarPergunta/config/settings.py` | Guarda configuracoes como modelo, URL do LM Studio, quantidade de perguntas, tipos e caminhos. |
| `gerarPergunta/services/knowledge_loader.py` | Le o arquivo de conhecimento e transforma as respostas em objetos `Exercicio`. |
| `gerarPergunta/models/exercicio.py` | Define a estrutura de um exercicio: numero, titulo e codigo do aluno. |
| `gerarPergunta/utils/sorteador.py` | Sorteia os tipos de pergunta sem repetir antes de usar todos os tipos disponiveis. |
| `gerarPergunta/services/question_orchestrator.py` | Coordena a geracao: escolhe exercicios, pega tipos sorteados e chama a IA. |
| `gerarPergunta/services/question_builder.py` | Monta o prompt correto para cada tipo de pergunta. |
| `gerarPergunta/services/ai_client.py` | Envia o prompt para o LM Studio e converte o JSON retornado em objetos `Pergunta`. |
| `gerarPergunta/models/pergunta.py` | Define a estrutura de uma pergunta gerada: tipo, texto e numero do exercicio original. |
| `gerarPergunta/services/report_exporter.py` | Formata e salva o resultado final em `perguntasGeradas.txt`. |

## Configuracoes

As configuracoes ficam em `gerarPergunta/config/settings.py`.

Principais valores:

```python
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
MODEL = "qwen/qwen3-vl-4b"
TOTAL_PERGUNTAS = 5
TEMPERATURE = 0.4
MAX_TOKENS = 512
```

O sistema usa a biblioteca `openai`, mas aponta para o servidor local do LM Studio. Isso significa que, para funcionar, o LM Studio precisa estar aberto, com um modelo carregado e expondo a API em `http://localhost:1234/v1`.

Tambem existem os tipos de pergunta:

```python
TIPOS_PERGUNTA = [
    "correcao",
    "justificativa",
    "descritiva",
    "modificacao",
    "previsao",
]
```

Com `TOTAL_PERGUNTAS = 5` e cinco tipos cadastrados, cada execucao tenta gerar cinco perguntas, normalmente uma de cada tipo, em ordem aleatoria.

Observacao: `MAX_TOKENS` esta configurado, mas atualmente nao e passado na chamada da API em `AIClient`. Na pratica, o limite de tokens depende do comportamento padrao do modelo/servidor.

## Entrada de dados

O arquivo de entrada contem os enunciados das questoes e as respostas dos alunos.

O formato esperado pelo `KnowledgeLoader` e este:

```text
1. Titulo da questao
2. Outro titulo

Resposta 1 -
codigo do aluno

Resposta 2 -
codigo do aluno
```

O carregador procura duas coisas:

- linhas de titulo no formato `numero. titulo`;
- blocos de resposta no formato `Resposta numero -`.

Internamente, ele usa estas expressoes regulares:

```python
_RE_QUESTAO = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*$")
_RE_RESPOSTA = re.compile(r"^\s*Resposta\s*(\d+)\s*-\s*$", re.IGNORECASE)
```

Isso tem uma consequencia importante: o titulo precisa estar com ponto depois do numero, por exemplo `1. Contagem de vogais`. Se o arquivo usar `1- Contagem de vogais`, o titulo nao sera reconhecido e o sistema usara um titulo generico, como `Questao 1`.

No estado atual do projeto, o arquivo `conteudo/conhecimento.txt` usa linhas como:

```text
1- Contagem de vogais em uma string
2 - Numeros primos em um intervalo
```

Por isso, se esse arquivo for usado sem ajuste, as respostas sao encontradas, mas os titulos podem cair no padrao `Questao 1`, `Questao 2`, etc.

## Como o arquivo e carregado

O processo comeca em `main.py`.

Primeiro, ele cria um `KnowledgeLoader`:

```python
loader = KnowledgeLoader(ARQUIVO_CONHECIMENTO)
exercicios = loader.carregar()
```

Dentro de `KnowledgeLoader.carregar()` acontece isto:

1. Verifica se o arquivo existe.
2. Le todas as linhas com UTF-8.
3. Separa titulos e respostas.
4. Monta uma lista de objetos `Exercicio`.

Cada `Exercicio` tem:

```python
numero: int
titulo: str
codigo: str
```

Se uma resposta estiver vazia, o modelo `Exercicio` substitui o codigo por:

```text
[O ALUNO NAO ESCREVEU CODIGO]
```

As respostas sao ordenadas pelo numero antes de virar lista. Isso deixa a ordem final previsivel: resposta 1, resposta 2, resposta 3, e assim por diante.

## Caminho do arquivo de entrada

Em `settings.py`, o caminho configurado e:

```python
ARQUIVO_CONHECIMENTO = "conhecimento.txt"
```

Esse caminho e relativo ao diretorio em que o comando Python for executado.

Ponto de atencao: no estado atual do repositorio, existe `conteudo/conhecimento.txt`, mas nao aparece um `conhecimento.txt` diretamente dentro de `gerarPergunta`. Entao, para executar sem erro, e preciso que o caminho configurado bata com a localizacao real do arquivo.

Exemplos de caminhos possiveis:

```python
ARQUIVO_CONHECIMENTO = "../conteudo/conhecimento.txt"
```

se o script for executado de dentro da pasta `gerarPergunta`, ou:

```python
ARQUIVO_CONHECIMENTO = "conteudo/conhecimento.txt"
```

se o script for executado da raiz do projeto.

## Sorteio dos tipos de pergunta

O sorteio fica em `gerarPergunta/utils/sorteador.py`.

O `Sorteador` recebe a lista de tipos e embaralha blocos completos. Isso significa que ele nao repete um tipo antes de todos os outros tipos terem aparecido.

Exemplo com cinco tipos:

```text
["correcao", "justificativa", "descritiva", "modificacao", "previsao"]
```

Uma execucao pode gerar:

```text
["previsao", "correcao", "descritiva", "modificacao", "justificativa"]
```

Outra execucao pode gerar:

```text
["justificativa", "modificacao", "previsao", "correcao", "descritiva"]
```

O algoritmo e:

1. Copia a lista de tipos.
2. Embaralha essa copia com `random.shuffle`.
3. Adiciona o bloco embaralhado ao resultado.
4. Repete ate atingir a quantidade pedida.
5. Corta a lista no tamanho exato solicitado.

Se `TOTAL_PERGUNTAS` for maior que a quantidade de tipos, o sistema cria outro bloco embaralhado. Por exemplo, com 8 perguntas e 5 tipos, ele usa os 5 tipos uma vez e depois sorteia mais 3 de um novo bloco.

Como nao existe uma semente fixa (`seed`), a ordem muda a cada execucao.

## Escolha dos exercicios usados

A classe `QuestionOrchestrator` decide quais exercicios entram em cada pergunta.

Ela recebe a lista carregada pelo `KnowledgeLoader` e monta uma lista de slots.

Se existem exercicios suficientes:

```python
return exercicios[:total]
```

Ou seja, com cinco exercicios e `TOTAL_PERGUNTAS = 5`, ele usa os cinco primeiros.

Se existem menos exercicios que perguntas desejadas, ele repete os exercicios em ciclo.

Exemplo: dois exercicios e cinco perguntas.

```text
slot 1 -> exercicio 1
slot 2 -> exercicio 2
slot 3 -> exercicio 1
slot 4 -> exercicio 2
slot 5 -> exercicio 1
```

Se a lista de exercicios estiver vazia, o sistema interrompe com:

```text
ValueError: Lista de exercicios esta vazia.
```

## Tipos de pergunta

Cada tipo tem uma intencao pedagogica diferente.

### correcao

Gera uma pergunta para levar o aluno a identificar possiveis erros logicos, sintaticos ou casos de borda no codigo.

Exemplo de objetivo:

```text
O aluno precisa olhar para o proprio codigo e encontrar um problema.
```

### justificativa

Pede que o aluno explique por que usou determinada abordagem.

Exemplo de objetivo:

```text
O aluno precisa justificar uma decisao de implementacao.
```

### descritiva

Pede que o aluno descreva o funcionamento do codigo passo a passo.

Exemplo de objetivo:

```text
O aluno precisa demonstrar compreensao do fluxo do programa.
```

### modificacao

Propoe uma mudanca concreta no codigo, como uma nova funcionalidade, refatoracao ou otimizacao.

Exemplo de objetivo:

```text
O aluno precisa adaptar o codigo para uma nova exigencia.
```

### previsao

Apresenta uma entrada especifica e pergunta qual sera a saida esperada.

Exemplo de objetivo:

```text
O aluno precisa simular mentalmente a execucao do codigo.
```

## Montagem do prompt

O prompt e montado em `QuestionBuilder`.

Cada tipo tem um metodo proprio:

```python
_correcao()
_justificativa()
_descritiva()
_modificacao()
_previsao()
```

Todos seguem a mesma estrutura:

1. Papel da IA: "Voce e um tutor de programacao."
2. Instrucao especifica do tipo de pergunta.
3. Cabecalho com numero, titulo e codigo do exercicio.
4. Regras obrigatorias.
5. Formato de resposta esperado em JSON.

O cabecalho inclui:

```text
Exercicio N: titulo

Codigo do aluno:
codigo...
```

O rodape obriga a IA a seguir estas regras:

```text
- Nao forneca a resposta
- Nao copie o codigo completo
- Seja claro e objetivo

Retorne APENAS um JSON valido, sem texto extra:
[{"tipo": "tipo", "pergunta": "<texto da pergunta>"}]
```

Essa parte e essencial, porque o `AIClient` espera receber JSON. Se a IA responder com texto solto, markdown quebrado ou JSON invalido, o parser pode descartar a resposta.

## Chamada da IA

A chamada acontece em `gerarPergunta/services/ai_client.py`.

O cliente e criado assim:

```python
self._client = OpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key=LM_STUDIO_API_KEY,
)
```

Apesar de usar a classe `OpenAI`, a requisicao vai para o LM Studio local, porque `base_url` aponta para:

```text
http://localhost:1234/v1
```

Para cada par `(exercicio, tipo)`, o sistema:

1. Monta o prompt com `QuestionBuilder`.
2. Chama `chat.completions.create`.
3. Envia uma mensagem de sistema dizendo que a IA e um tutor de programacao.
4. Envia o prompt como mensagem de usuario.
5. Usa o modelo configurado em `MODEL`.
6. Usa a temperatura configurada em `TEMPERATURE`.

Trecho central:

```python
response = self._client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "Voce e um tutor de programacao."},
        {"role": "user", "content": prompt},
    ],
    temperature=TEMPERATURE,
)
```

Temperatura `0.4` deixa a geracao relativamente controlada, mas ainda com alguma variacao. Isso combina com o objetivo do projeto: gerar perguntas diferentes sem perder muito foco.

## Cache das chamadas

O `AIClient` usa `lru_cache` no metodo `_chamar_api_cached`.

A chave do cache considera:

```python
(numero, titulo, codigo, tipo)
```

Isso ajuda quando o mesmo exercicio e o mesmo tipo aparecem de novo. Nesse caso, dentro da mesma instancia do cliente, o sistema pode reutilizar o resultado em vez de chamar a IA novamente.

Esse cache nao salva nada em arquivo. Ele dura apenas enquanto o programa esta rodando.

## Conversao da resposta da IA

Depois que a IA responde, o sistema pega:

```python
response.choices[0].message.content.strip()
```

Em seguida, tenta extrair um array JSON.

A funcao `_extrair_json()` procura:

- o primeiro caractere `[`;
- o ultimo caractere `]`;
- o conteudo entre eles.

Depois chama:

```python
json.loads(...)
```

Se der certo, o sistema espera uma lista de dicionarios parecida com:

```json
[
  {
    "tipo": "descritiva",
    "pergunta": "Descreva o que o codigo faz passo a passo."
  }
]
```

Cada item vira um objeto `Pergunta`:

```python
Pergunta(
    tipo=tipo,
    pergunta=pergunta,
    exercicio_numero=exercicio_numero,
)
```

Se o JSON nao for encontrado ou estiver invalido, o sistema mostra um aviso e retorna lista vazia. Isso significa que uma falha da IA pode fazer o arquivo final sair com menos perguntas que o total pedido.

## Exportacao do resultado

A exportacao fica em `ReportExporter`.

O `main.py` chama:

```python
exporter = ReportExporter(ARQUIVO_SAIDA)
exporter.exportar(pares)
```

O caminho configurado e:

```python
ARQUIVO_SAIDA = "../conteudo/perguntasGeradas.txt"
```

O exportador:

1. Recebe a lista de pares `(Exercicio, Pergunta)`.
2. Monta um texto formatado.
3. Cria o diretorio de destino, se ele nao existir.
4. Escreve o arquivo final com UTF-8.

O formato gerado por bloco e:

```text
============================================================
Exercicio gerado com base na sua resposta da questao N: Titulo
============================================================

1 - [TIPO] Texto da pergunta

Seu codigo:
----------------------------------------
codigo do aluno
----------------------------------------
```

Ponto importante: o `ReportExporter` nao gera respostas para as perguntas. Ele salva a pergunta e o codigo de referencia. Se `conteudo/perguntasGeradas.txt` tiver secoes como `resposta 1 -`, essas respostas foram adicionadas depois ou vieram de outro processo, nao deste exportador.

Outro ponto importante: `write_text` sobrescreve o arquivo. Entao, a cada nova execucao, o conteudo anterior de `perguntasGeradas.txt` pode ser substituido.

## Exemplo completo com cinco exercicios

Com `TOTAL_PERGUNTAS = 5` e cinco respostas no arquivo de conhecimento, o fluxo fica assim:

1. `KnowledgeLoader` encontra cinco blocos `Resposta 1 -` ate `Resposta 5 -`.
2. Ele cria cinco objetos `Exercicio`.
3. `QuestionOrchestrator` monta cinco slots, um para cada exercicio.
4. `Sorteador` embaralha os cinco tipos de pergunta.
5. Para cada slot, o sistema combina um exercicio com um tipo.
6. `QuestionBuilder` monta o prompt adequado.
7. `AIClient` envia o prompt ao LM Studio.
8. A IA retorna um JSON com a pergunta.
9. O JSON vira objeto `Pergunta`.
10. `ReportExporter` grava o resultado em `perguntasGeradas.txt`.

Exemplo de combinacao possivel:

```text
Exercicio 1 -> justificativa
Exercicio 2 -> correcao
Exercicio 3 -> descritiva
Exercicio 4 -> previsao
Exercicio 5 -> modificacao
```

Na proxima execucao, a ordem dos tipos pode mudar.

## Como adicionar um novo tipo de pergunta

Para adicionar um novo tipo, o sistema precisa de duas alteracoes.

Primeiro, adicionar o nome em `TIPOS_PERGUNTA`:

```python
TIPOS_PERGUNTA = [
    "correcao",
    "justificativa",
    "descritiva",
    "modificacao",
    "previsao",
    "novo_tipo",
]
```

Depois, adicionar um metodo em `QuestionBuilder` e incluir esse metodo no dicionario de `construir()`.

Exemplo:

```python
def _novo_tipo(self, exercicio: Exercicio) -> str:
    return (
        "Voce e um tutor de programacao.\n\n"
        "Gere 1 pergunta do tipo NOVO_TIPO ...\n\n"
        + self._cabecalho(exercicio)
        + self._rodape("novo_tipo")
    )
```

E no dicionario:

```python
metodos = {
    "correcao": self._correcao,
    "justificativa": self._justificativa,
    "descritiva": self._descritiva,
    "modificacao": self._modificacao,
    "previsao": self._previsao,
    "novo_tipo": self._novo_tipo,
}
```

## Analise dos pontos de atencao

### 1. Caminho do arquivo de conhecimento

O sistema procura `conhecimento.txt`, mas o arquivo encontrado no projeto esta em `conteudo/conhecimento.txt`.

Se o caminho nao for ajustado ou o arquivo nao for colocado no local esperado, a execucao termina com `FileNotFoundError`.

### 2. Formato dos titulos

O parser espera titulo com ponto:

```text
1. Titulo
```

Mas o arquivo atual usa hifen:

```text
1- Titulo
```

Com isso, as respostas ainda podem ser lidas, mas os titulos nao sao associados corretamente. O sistema usa `Questao N` como fallback.

### 3. A IA pode gerar menos perguntas

Se a chamada falhar, se o LM Studio nao estiver aberto ou se a IA devolver JSON invalido, o sistema retorna lista vazia para aquela tentativa.

Resultado: o arquivo final pode ter menos blocos do que `TOTAL_PERGUNTAS`.

### 4. O modelo nao e validado semanticamente

O parser confere se existe JSON valido, mas nao garante que a pergunta seja pedagogicamente boa, que o tipo retornado seja exatamente o tipo pedido ou que a pergunta nao contenha resposta escondida.

Essa qualidade depende principalmente do prompt e do modelo usado.

### 5. `MAX_TOKENS` nao esta sendo usado

A configuracao existe, mas nao entra na chamada `chat.completions.create`.

Se for necessario limitar o tamanho da resposta, seria preciso passar:

```python
max_tokens=MAX_TOKENS
```

na chamada da API.

### 6. A saida e sobrescrita

Cada execucao salva novamente em `perguntasGeradas.txt`. O arquivo nao acumula historico automaticamente.

### 7. O gerador cria perguntas, nao correcoes

O objetivo deste modulo e gerar perguntas para o aluno responder. Ele nao avalia as respostas, nao da nota e nao gera correcao automatica. Essa responsabilidade pertence ao modulo `correcao`.

## Resumo final

O `gerarPergunta` funciona como uma esteira:

1. Le codigos dos alunos.
2. Transforma cada resposta em um objeto `Exercicio`.
3. Escolhe tipos de pergunta de forma embaralhada.
4. Monta um prompt especifico para cada tipo.
5. Chama uma IA local pelo LM Studio.
6. Extrai o JSON retornado.
7. Converte o JSON em objetos `Pergunta`.
8. Salva as perguntas geradas em um arquivo de texto.

A parte mais importante da geracao esta no prompt. O codigo organiza o processo, mas quem escreve a pergunta final e o modelo de IA. Por isso, a qualidade final depende de tres coisas: o codigo do aluno estar bem carregado, o tipo sorteado estar correto e a IA obedecer ao formato JSON solicitado.
