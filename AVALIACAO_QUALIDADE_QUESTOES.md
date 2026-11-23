# Avaliação da qualidade das questões geradas

> **Pergunta do orientador:** "Como você pretende avaliar a qualidade das questões: com quais critérios e com quem avaliando?"

---

## 1. Resposta resumida

A qualidade das questões será avaliada em **duas camadas complementares**:

1. **Verificação automática**, feita pelo próprio protótipo. Cobre os aspectos objetivos: o código-base executa, os dados de entrada são válidos, a questão segue o formato esperado e pode ser corrigida automaticamente.
2. **Avaliação por especialistas**, feita por **três avaliadores com experiência no ensino de programação introdutória** (o orientador e dois professores ou monitores da área). Eles aplicam, de forma **independente**, uma **rubrica de sete critérios** em escala de 0 a 2, mais um julgamento global: a questão "pode ser usada como está?".

A confiabilidade da avaliação humana será medida pela **concordância entre avaliadores (Kappa de Fleiss)**. Os resultados serão apresentados por critério, por tipo de questão (PREVISAO, MODIFICACAO, DESCRITIVA, JUSTIFICATIVA, CORRECAO) e por nível do experimento (básico, médio, avançado).

Esse desenho segue o que é mais comum na literatura de geração automática de questões. Kurdi et al. (2020) mostram que a maioria dos trabalhos da área avalia as questões com juízes especialistas e critérios como relevância, clareza, dificuldade e respondibilidade. Sarsa et al. (2022), que geraram exercícios de programação com LLMs, combinaram avaliação manual com verificação automática por execução de código e testes.

---

## 2. O que exatamente está sendo avaliado

No protótipo, uma **questão gerada** não é só um enunciado. Ela é composta por:

| Componente | Exemplo no sistema |
|---|---|
| Tipo da questão | `[PREVISAO]`, `[MODIFICACAO]`, `[DESCRITIVA]`, `[JUSTIFICATIVA]`, `[CORRECAO]` |
| Enunciado | Texto gerado pelo LLM |
| Código-base | Código anterior do aluno, extraído de `conhecimento.txt` |
| Dados de entrada / testes | `entradaTestes`, casos de teste usados pela correção |

Por isso, a avaliação considera a questão **como um todo**: um enunciado claro com um código-base que não executa, ou com testes incoerentes, **não** é uma questão de qualidade.

**Fora do escopo desta avaliação:** a qualidade da *correção* das respostas dos alunos (avaliada à parte) e o efeito das questões na *aprendizagem*. Medir aprendizagem exigiria um estudo com alunos reais e grupo de controle, que não cabe no prazo do TCC (ver Seção 8).

---

## 3. Critérios de avaliação (rubrica)

### 3.1 Escala

Cada critério recebe uma nota em **escala de 3 níveis**:

- **2: atende** plenamente
- **1: atende parcialmente** (precisa de pequena edição)
- **0: não atende**

A escala de 3 níveis foi escolhida em vez de uma Likert de 5 pontos porque descritores mais simples tendem a gerar **maior concordância entre avaliadores** e são mais fáceis de justificar. Cada nível tem um descritor objetivo, na tabela a seguir.

### 3.2 Critérios

| # | Critério | Pergunta que o avaliador responde | 2: atende | 1: parcial | 0: não atende | Quem verifica |
|---|---|---|---|---|---|---|
| **C1** | **Correção técnica** | O enunciado e o código-base estão tecnicamente corretos? | Sem erros conceituais; código executa (exceto quando o erro é proposital, como em CORRECAO) | Pequena imprecisão que não impede a resolução | Erro que impede ou distorce a resolução | Automático + especialista |
| **C2** | **Clareza e não ambiguidade** | O aluno entende o que deve fazer, com uma única interpretação razoável? | Claro e inequívoco | Entendível, mas com termos vagos ou mais de uma leitura possível | Confuso ou contraditório | Especialista |
| **C3** | **Aderência ao tipo** | A questão realmente exige a atividade do tipo declarado? | Exige exatamente a atividade do tipo (prever, modificar, descrever, justificar, corrigir) | Mistura tipos ou exige pouco da atividade declarada | Não corresponde ao tipo, ou **já entrega a resposta** | Especialista |
| **C4** | **Vínculo com o código do aluno** | A questão parte do que *este* aluno escreveu? | Explora algo específico do código do aluno (uma decisão, um erro, uma estrutura) | Usa o código só como pano de fundo; serviria para qualquer aluno | Sem relação com o código do aluno | Especialista |
| **C5** | **Relevância pedagógica** | A questão exercita um conceito importante de programação introdutória? | Trabalha um conceito central ou um erro relevante presente no código | Conceito secundário; ignora um problema mais importante visível no código | Trivial ou irrelevante para o objetivo do exercício | Especialista |
| **C6** | **Adequação da dificuldade** | A dificuldade é compatível com o nível do exercício de origem? | Compatível | Um pouco fácil ou difícil demais | Muito desproporcional | Especialista |
| **C7** | **Respondibilidade e corrigibilidade** | Dá para responder com as informações dadas e corrigir de forma confiável? | Informação suficiente; testes e entradas coerentes com o enunciado | Faltam detalhes, ou parte dos testes é incoerente | Impossível responder ou corrigir corretamente | Automático + especialista |

### 3.3 Julgamento global

Além dos sete critérios, cada avaliador responde:

> **A questão pode ser usada com alunos?**
> ( ) Sim, como está  ( ) Sim, com pequenas edições  ( ) Não

Esse indicador, de "prontidão para uso", é o mais fácil de comunicar na banca ("X% das questões puderam ser usadas sem alteração"). É semelhante ao critério usado por Sarsa et al. (2022) para exercícios gerados por LLM.

### 3.4 Relação com a Taxonomia de Bloom

Como apoio aos critérios C5 e C6, cada questão também pode ser **classificada** (sem nota) no nível cognitivo da Taxonomia de Bloom revisada (ANDERSON; KRATHWOHL, 2001). Os tipos do sistema correspondem a níveis diferentes:

| Tipo | Nível de Bloom predominante |
|---|---|
| PREVISAO | Compreender / Aplicar |
| DESCRITIVA | Compreender |
| MODIFICACAO | Aplicar / Criar |
| CORRECAO | Analisar |
| JUSTIFICATIVA | Avaliar |

Isso permite verificar se o sistema produz questões de níveis cognitivos variados, e não apenas de memorização.

---

## 4. Quem avalia

### 4.1 Camada 1: verificação automática (o próprio sistema)

Esses itens são objetivos e não dependem de opinião. O protótipo já tem a infraestrutura para eles (execução isolada em `correcao/execution/runner.py`, validação do contrato de geração e testes em `correcao/tests/generator.py`):

| Verificação | Critério relacionado |
|---|---|
| O código-base executa sem erro (exceto CORRECAO, cujo erro é proposital) | C1 |
| A questão segue o formato/contrato esperado (tipo reconhecido, campos presentes) | C3, C7 |
| Os dados de entrada (`entradaTestes`) são compatíveis com o código-base | C7 |
| Em PREVISAO, a saída esperada é obtida executando o código, e não "adivinhada" pelo LLM | C1, C7 |
| A solução de referência passa nos casos de teste | C7 |

O resultado é uma **taxa de questões válidas automaticamente**, apresentada antes da avaliação humana.

### 4.2 Camada 2: avaliação por especialistas

**Perfil dos avaliadores (3 pessoas):**
- o orientador;
- dois professores, ou monitores de nível avançado, que já atuaram em disciplinas de algoritmos ou programação introdutória.

**Critério de inclusão:** ter ministrado ou monitorado ao menos um semestre de disciplina introdutória de programação. O texto do TCC deve relatar a experiência de cada avaliador, sem identificá-los se preferirem.

**Por que três:** com dois avaliadores, um desacordo não tem desempate; com três, é possível usar a **mediana** ou a **maioria** como nota final de cada critério, e calcular o Kappa de Fleiss.

**Procedimento:**

1. **Calibração (cerca de 15 min):** os avaliadores recebem a rubrica e avaliam juntos 2 questões de exemplo que **não** entram na amostra. As divergências são discutidas para alinhar a interpretação dos descritores.
2. **Avaliação independente:** cada avaliador preenche a ficha (Apêndice A) para todas as questões da amostra, **sem acesso às notas dos outros** e **sem ver as respostas dos alunos nem a correção automática**, para não ser influenciado pelo desempenho do aluno.
3. **Ordem aleatória:** as questões são apresentadas em ordem embaralhada, para evitar que o avaliador perceba padrões por nível ou por tipo.
4. **Consolidação:** a nota final de cada critério é a mediana dos três avaliadores. Comentários livres são reunidos para a análise qualitativa.

**Esforço estimado:** cerca de 25 questões × 2 a 3 min por questão ≈ **1 hora por avaliador**. É viável dentro do cronograma do TCC.

### 4.3 Camada 3 (opcional): percepção dos alunos

Se houver tempo, um pequeno grupo de alunos de disciplina introdutória pode responder às questões e preencher um questionário curto sobre **clareza** e **dificuldade percebida** (escala de 1 a 5). Isso **não substitui** os especialistas: mede a percepção do público-alvo, e não a qualidade técnica.

---

## 5. Amostra

As questões avaliadas serão as geradas nos experimentos do protótipo (pasta `experimento/`). Situação atual:

| Experimento | Questões geradas | Tipos cobertos |
|---|---|---|
| `basico/` | 5 | PREVISAO, MODIFICACAO, DESCRITIVA, JUSTIFICATIVA, CORRECAO |
| `medio/` | 5 | os mesmos 5 tipos |
| `listaRodack/respondidos/` | 5 | os mesmos 5 tipos |
| `listaRodack/naoRespondidos/` | 4 | CORRECAO, DESCRITIVA, JUSTIFICATIVA, MODIFICACAO |
| `avancado/` | **0** (arquivo ainda vazio) | — |
| **Total atual** | **19** | |

**Pendência:** gerar as questões do experimento `avancado/` (já existe `conhecimento3.txt`). Com isso, a amostra chega a **cerca de 24 questões**, cobrindo os **5 tipos nos 3 níveis**. É uma amostra pequena, adequada a um estudo exploratório de TCC, e permite analisar todas as questões em vez de sortear uma parte.

---

## 6. Análise dos resultados

### 6.1 Métricas quantitativas

| Métrica | Como calcular |
|---|---|
| **Taxa de validade automática** | % de questões aprovadas em todas as verificações da Camada 1 |
| **Nota média por critério** | Média das notas consolidadas (0 a 2) de cada critério |
| **% de questões que atendem cada critério** | % com nota consolidada = 2 |
| **Prontidão para uso** | % "usável como está", % "com pequenas edições", % "não usável" |
| **Resultados por tipo e por nível** | As métricas acima separadas por PREVISAO/MODIFICACAO/… e por básico/médio/avançado |
| **Concordância entre avaliadores** | **Kappa de Fleiss** (FLEISS, 1971) por critério |

**Interpretação do Kappa** (LANDIS; KOCH, 1977):

| κ | Concordância |
|---|---|
| < 0,00 | Pobre |
| 0,00 a 0,20 | Leve |
| 0,21 a 0,40 | Razoável |
| 0,41 a 0,60 | Moderada |
| 0,61 a 0,80 | Substancial |
| 0,81 a 1,00 | Quase perfeita |

Um critério com Kappa baixo indica que a **rubrica estava ambígua naquele ponto**. Isso deve ser discutido como limitação, e não escondido.

> Implementação: o Kappa de Fleiss pode ser calculado em Python com `statsmodels.stats.inter_rater.fleiss_kappa`.

### 6.2 Análise qualitativa

Os comentários dos avaliadores e as questões com nota 0 serão agrupados em **categorias de defeito**, por exemplo: "entrega a resposta no enunciado", "ignora o erro principal do código", "testes incoerentes", "ambiguidade". O objetivo é responder **por que** o gerador falha e **em quais tipos** isso acontece mais. Isso orienta melhorias nos prompts.

---

## 7. Aplicação ilustrativa da rubrica (pré-teste do autor)

Para verificar se os critérios **conseguem discriminar** questões boas de ruins, o autor aplicou a rubrica às questões do experimento `basico/`. **Esta não é a avaliação final:** é um pré-teste da rubrica, que será refeito pelos especialistas.

| Q | Tipo | Observação | Critérios afetados |
|---|---|---|---|
| 1 | PREVISAO | Enunciado claro, com entrada explícita. Porém `entradaTestes` (`"7.0\n8.5\n6.0\n10.0\n"`) não corresponde ao código-base, que não lê `input()` e já chama a função com a lista fixa. | C7 = 1 |
| 2 | MODIFICACAO | Pede para também retornar a lista de pares, mas **não aborda o erro real do aluno** (`range(1, len(lista))` ignora o índice 0). O aluno manteve o erro na resposta. | C4 = 1, C5 = 1 |
| 3 | DESCRITIVA | Clara e coerente com o tipo, mas genérica. Não induz o aluno a pensar no caso de empate, que é justamente onde o código dele falha. | C4 = 1, C5 = 1 |
| 4 | JUSTIFICATIVA | Bom vínculo com uma decisão do aluno (uso de `while`) e **revelou uma concepção errada** ("o `for` só funciona para listas e strings"). Porém ignora o problema mais grave do código (`print` em vez de `return`). | C4 = 2, C5 = 1 |
| 5 | CORRECAO | O enunciado **já entrega a resposta** ("falta o ':' no final da linha"). Assim, a questão deixa de exigir que o aluno encontre o erro. | C3 = 0 |

**O que o pré-teste mostra:**
- A rubrica consegue **diferenciar** questões: a Q4 é pedagogicamente rica, enquanto a Q5 falha no próprio objetivo do tipo.
- Os critérios C4 e C5 capturam um padrão importante: **o gerador nem sempre direciona a pergunta ao erro principal do aluno**. Esse é um achado útil para a discussão do TCC.
- A Camada 1 (automática) pegaria o problema de C7 da Q1, mas não o de C3 da Q5. Isso **justifica a necessidade de avaliadores humanos**.

---

## 8. Limitações e ameaças à validade

Declarar as limitações fortalece o trabalho na banca:

- **Amostra pequena (cerca de 24 questões):** os resultados são exploratórios e não permitem generalização estatística.
- **Proximidade dos avaliadores com o autor:** o orientador conhece o projeto. Isso é mitigado pela avaliação independente, pela rubrica com descritores e pela inclusão de avaliadores externos ao projeto.
- **Avaliadores sabem que as questões foram geradas por IA:** isso pode gerar viés, positivo ou negativo. Uma melhoria futura seria misturar questões escritas por professores, sem identificar a origem, e comparar as duas.
- **Não se mede aprendizagem:** a avaliação trata da qualidade das questões como artefato. Verificar se elas melhoram o aprendizado exigiria um estudo com alunos e grupo de controle, deixado como trabalho futuro.
- **Variabilidade do LLM:** o mesmo conhecimento pode gerar questões diferentes a cada execução. As questões avaliadas devem ser **congeladas** (arquivos versionados no repositório), para que o resultado seja reprodutível.

---

## 9. Resposta pronta para o orientador

> "Vou avaliar as questões em duas camadas. Na primeira, o próprio sistema verifica de forma automática os aspectos objetivos: se o código-base executa, se os dados de entrada e os testes são coerentes e se a questão segue o formato esperado. Na segunda, três avaliadores com experiência no ensino de programação introdutória (o senhor e mais dois professores ou monitores) aplicam de forma independente uma rubrica de sete critérios: correção técnica, clareza, aderência ao tipo, vínculo com o código do aluno, relevância pedagógica, adequação da dificuldade e corrigibilidade. Cada critério recebe nota de 0 a 2, e há ainda um julgamento global de se a questão pode ser usada com alunos. Vou medir a concordância entre os avaliadores com o Kappa de Fleiss e apresentar os resultados por critério, por tipo de questão e por nível. A amostra são as cerca de 24 questões geradas nos experimentos básico, médio, avançado e na lista de algoritmos. Fiz um pré-teste da rubrica nas questões do experimento básico, e ela já identificou problemas reais, como uma questão de correção que entregava a resposta no próprio enunciado."

---

## 10. Próximos passos

- [ ] Gerar as questões do experimento `avancado/`.
- [ ] Congelar (versionar) os arquivos de questões que serão avaliados.
- [ ] Rodar a verificação automática (Camada 1) e registrar os resultados.
- [ ] Convidar os dois avaliadores além do orientador.
- [ ] Preparar a ficha de avaliação (planilha ou formulário) com as questões em ordem aleatória.
- [ ] Sessão de calibração com 2 questões fora da amostra.
- [ ] Coletar as avaliações, calcular as métricas e o Kappa de Fleiss.
- [ ] Escrever as seções de resultados e discussão.

---

## Apêndice A: Ficha de avaliação (por questão)

```
Questão nº: ____     Avaliador: ____

Leia o enunciado, o código-base e os dados de entrada. Marque uma opção por critério.
(0 = não atende | 1 = atende parcialmente | 2 = atende)

C1  Correção técnica ........................ [0] [1] [2]
C2  Clareza e não ambiguidade ............... [0] [1] [2]
C3  Aderência ao tipo declarado ............. [0] [1] [2]
C4  Vínculo com o código do aluno ........... [0] [1] [2]
C5  Relevância pedagógica ................... [0] [1] [2]
C6  Adequação da dificuldade ................ [0] [1] [2]
C7  Respondibilidade e corrigibilidade ...... [0] [1] [2]

Nível de Bloom predominante:
[ ] Lembrar [ ] Compreender [ ] Aplicar [ ] Analisar [ ] Avaliar [ ] Criar

A questão pode ser usada com alunos?
[ ] Sim, como está   [ ] Sim, com pequenas edições   [ ] Não

Comentário (obrigatório se algum critério = 0):
______________________________________________________________
```

---

## Referências

> Confira cada referência (volume, páginas, DOI) na fonte original antes de incluí-la no TCC.

ANDERSON, L. W.; KRATHWOHL, D. R. (ed.). **A taxonomy for learning, teaching, and assessing**: a revision of Bloom's taxonomy of educational objectives. New York: Longman, 2001.

FLEISS, J. L. Measuring nominal scale agreement among many raters. **Psychological Bulletin**, v. 76, n. 5, p. 378-382, 1971.

KURDI, G.; LEO, J.; PARSIA, B.; SATTLER, U.; AL-EMARI, S. A systematic review of automatic question generation for educational purposes. **International Journal of Artificial Intelligence in Education**, v. 30, n. 1, p. 121-204, 2020.

LANDIS, J. R.; KOCH, G. G. The measurement of observer agreement for categorical data. **Biometrics**, v. 33, n. 1, p. 159-174, 1977.

SARSA, S.; DENNY, P.; HELLAS, A.; LEINONEN, J. Automatic generation of programming exercises and code explanations using large language models. In: ACM CONFERENCE ON INTERNATIONAL COMPUTING EDUCATION RESEARCH, 18., 2022, Lugano. **Proceedings** [...]. New York: ACM, 2022. p. 27-43.
