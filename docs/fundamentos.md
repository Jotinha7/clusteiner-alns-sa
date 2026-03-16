# Fundamentos: Problema de Árvore de Steiner Clusterizada (CluSteiner)

Este capítulo descreve os conceitos básicos necessários para entender o problema que o TCC aborda e a heurística de referência usada no artigo base.

---

## 1. Definições formais

### 1.1 Grafo e custos

Trabalhamos com um grafo não-direcionado e ponderado $G = (V, E)$, onde:

- $V$: conjunto de vértices;
- $E$: conjunto de arestas não dirigidas $\{u, v\}$ com $u, v \in V$.

Cada aresta $e = \{u, v\} \in E$ possui um custo $c_e > 0$. Em instâncias euclidianas, esse custo é derivado da distância euclidiana entre as coordenadas dos vértices $u$ e $v$.

### 1.2 Vértices terminais e Steiner

O conjunto de vértices é particionado em:

- **Vértices terminais** $R \subseteq V$: nós que devem obrigatoriamente aparecer na solução.
- **Vértices Steiner** $S = V \setminus R$: nós opcionais, que podem ser usados para “baratear” a árvore (diminuir o custo total), mas não são obrigatórios.

> **Intuição:** terminais são pontos que precisam estar conectados (por exemplo, clientes ou estações); vértices Steiner são pontos intermediários que podem aparecer na árvore se valer a pena.

### 1.3 Clusters de terminais

No problema *Clustered Steiner Tree*, o conjunto de terminais $R$ é particionado em clusters. Definimos o conjunto de clusters $\mathcal{C} = \{R_1, R_2, \dots, R_h\}$ tal que:

- $R_1 \cup R_2 \cup \dots \cup R_h = R$;
- $R_i \cap R_j = \emptyset$ para $i \neq j$.

Cada $R_k$ é um cluster de terminais. Em todas as instâncias do dataset usado neste TCC, todos os vértices de $R$ são terminais obrigatórios, ou seja, a solução precisa conter todos os vértices de todos os clusters.

### 1.4 Árvore de Steiner clusterizada

Uma solução viável é uma árvore $T = (V_T, E_T)$ tal que:

1. $T$ é uma árvore (conectado e sem ciclos);
2. $R \subseteq V_T \subseteq V$;
3. $E_T \subseteq E$.

O custo da solução é dado por:

$$
\text{custo}(T) = \sum_{e \in E_T} c_e
$$

O problema de Árvore de Steiner Clusterizada (CluSteiner) é definido como:

> Encontrar uma árvore $T$ de menor custo que contenha todos os vértices terminais $R$ (agrupados em clusters), podendo usar vértices Steiner $S$ opcionalmente.

---

## 2. Restrição de disjunção no CluSteiner

Nesta seção vamos reescrever a **restrição de disjunção** exatamente na forma em que o problema é definido no artigo base do CluSteiner e, em seguida, mostrar como ela aparece na modelagem matemática que vamos usar no TCC.

### 2.1 Definição no artigo base

Considere um grafo ponderado e não-direcionado $G = (V, E, w)$, onde $V$ é o conjunto de vértices, $E$ é o conjunto de arestas e $w : E \to \mathbb{R}_{+}$ é a função de pesos.

Seja $R \subseteq V$ o conjunto de **vértices requeridos** (terminais). O artigo define ainda uma partição de $R$ em $k$ clusters:

$$
R' = \{R_1, R_2, \dots, R_k\}, \quad R_i \subseteq R, \quad R_i \cap R_j = \emptyset \ (i \neq j), \quad \bigcup_{i=1}^k R_i = R.
$$

Uma solução do CluSteiner é uma árvore de Steiner $T = (V_T, E_T)$ tal que:

**1. Condição de Steiner tree**

Todos os vértices requeridos aparecem na árvore:

$$
R \subseteq V_T
$$

**2. Árvore local de cada cluster**

Para cada cluster $R_i$, define-se a **árvore local** $T_i$ como o menor subgrafo de $T$ que ainda conecta todos os vértices de $R_i$:

$$
T_i = (V_{T_i}, E_{T_i}) \subseteq T
$$

com

$$
R_i \subseteq V_{T_i} \subseteq V_T
$$

**3. Restrição de exclusão (disjunção) entre árvores locais**

O artigo impõe que as árvores locais sejam **mutuamente disjuntas**:

$$
V_{T_i} \cap V_{T_j} = \emptyset, \qquad \forall\, 1 \le i < j \le k.
$$

Em palavras:
> Nenhum vértice pode pertencer a mais de uma árvore local ao mesmo tempo.

Isso vale tanto para vértices de Steiner quanto, indiretamente, para a forma como os clusters se conectam. Cada vértice (especialmente os de Steiner) é “alocado” a **no máximo um cluster** na solução final.

Essa é a forma em que a *restrição de disjunção* aparece na formulação do artigo: ela é embutida na frase “todas as árvores locais são mutuamente exclusivas”.

### 2.2 Regra de disjunção na modelagem do TCC

Para usar essa ideia em um modelo de Programação Inteira/MIP, introduzimos variáveis binárias que marcam em qual árvore local cada vértice está.

Para cada cluster $i \in \{1, \dots, k\}$ e para cada vértice $v \in V$, definimos:

$$
y_{i,v} = 
\begin{cases} 
1, & \text{se o vértice } v \text{ pertence à árvore local } T_i, \\
0, & \text{caso contrário.}
\end{cases}
$$

Com essas variáveis, a restrição de que **cada vértice de Steiner pertence a no máximo uma árvore local** é escrita como:

$$
\sum_{i=1}^{k} y_{i,v} \le 1, \qquad \forall\, v \in V \setminus R. \quad \text{(Disjunção-1)}
$$

**Interpretando:**

Para cada vértice $v$ que **não** é terminal (logo, $v \in V \setminus R$), as variáveis $\{y_{1,v}, y_{2,v}, \dots, y_{k,v}\}$ não podem ser 1 ao mesmo tempo.

Em lógica, isso significa:

> O vértice $v$ está na árvore local do cluster 1 **ou** na do cluster 2 **ou** ... **ou** não está em nenhuma árvore local.

**Para os vértices terminais, a regra é diferente:**

Se $r \in R_i$ é um terminal do cluster $i$, então ele **deve** estar na árvore local $T_i$:

$$
y_{i,r} = 1, \qquad \forall\, r \in R_i. \quad \text{(Disjunção-2)}
$$

( Opcionalmente, podemos forçar $y_{j,r} = 0$ para $j \neq i$, mas isso já é consequência da restrição Disjunção-1 se considerarmos que o terminal só pode pertencer ao seu próprio cluster. )

Juntas, (Disjunção-1) e (Disjunção-2) codificam, em forma linear, a mesma ideia da definição do artigo:

1.  Todos os vértices de um cluster aparecem em sua árvore local;
2.  Nenhum vértice é compartilhado entre árvores locais de clusters diferentes.

Em resumo, a **restrição de disjunção** do TCC será representada pelas seguintes condições:

$$
\begin{aligned}
& y_{i,v} \in \{0,1\}, && \forall i = 1,\dots,k,\ \forall v \in V, \\
& y_{i,r} = 1, && \forall i = 1,\dots,k,\ \forall r \in R_i, \\
& \sum_{i=1}^{k} y_{i,v} \le 1, && \forall v \in V \setminus R.
\end{aligned}
$$

Essas equações são a forma “oficial” da restrição de disjunção na formulação matemática do TCC, coerente com a definição de que as árvores locais $T_i$ devem ser mutuamente exclusivas.

---

## 3. Heurística 2-níveis: SPH + MST

O artigo base utiliza uma heurística em dois níveis para construir soluções iniciais para o problema CluSteiner. A ideia geral é:

1. **Nível local (dentro dos clusters):** conectar vértices terminais de forma barata usando uma heurística tipo “Shortest Path Heuristic” (SPH);
2. **Nível global (entre clusters):** tratar cada cluster (ou representante de cluster) como um super-nó e conectar esses super-nós usando uma heurística de Árvore Geradora Mínima (MST).

Isso gera uma árvore em duas camadas: dentro de cada cluster e entre clusters.

### 3.1 Shortest Path Heuristic (SPH) – visão simples

A *Shortest Path Heuristic* é uma família de heurísticas baseada em caminhos mínimos. Uma forma simplificada de explicar, adequada para a seção de Fundamentos, é:

> Começamos com um terminal qualquer e, iterativamente, conectamos o próximo terminal usando o caminho mais barato (soma de arestas) entre o terminal já conectado e algum terminal ainda desconectado.

Uma versão ilustrativa (não necessariamente igual à do artigo) é:

1. Escolha um terminal inicial $t_0$ e inicie a árvore $T$ com apenas esse vértice.
2. Enquanto existir um terminal ainda não contido em $T$:
   1. para cada terminal $t$ ainda desconectado, compute o menor caminho (no grafo original) entre $t$ e qualquer vértice já em $T$;
   2. escolha o terminal $t^*$ cujo caminho mínimo tem menor custo;
   3. adicione todas as arestas desse caminho em $T$.
3. Ao final, $T$ conecta todos os terminais (pode conter vértices Steiner).

### 3.2 Nível local: SPH dentro de cada cluster

Para aproveitar a estrutura em clusters, podemos aplicar SPH “cluster a cluster”:

- Para cada cluster $R_k$:
  - aplicamos a heurística SPH restrita aos terminais de $R_k$ (e, se necessário, vértices Steiner próximos);
  - obtém-se uma pequena árvore $T_k$ que conecta todos os vértices de $R_k$.

O resultado é um conjunto de árvores locais $T_1, T_2, \dots, T_h$. Cada uma conecta internamente os terminais de um cluster.

### 3.3 Nível global: MST entre clusters

Depois de conectar localmente cada cluster, o segundo nível enxerga cada cluster como um super-nó.

Uma forma simples de descrever é:

1. Para cada cluster $R_k$, escolher um vértice “representante” (por exemplo, o vértice de menor índice, ou o mais central). Denote esse conjunto de representantes por $Q = \{q_1, q_2, \dots, q_h\}$.

2. Definir um grafo reduzido $G' = (Q, E')$, onde o custo entre dois representantes $q_i$ e $q_j$ é dado pela menor distância entre qualquer vértice do cluster $i$ e qualquer vértice do cluster $j$ no grafo original.

3. Construir uma Árvore Geradora Mínima (MST) em $G'$, por exemplo com Prim ou Kruskal. Esse MST conecta todos os clusters com custo aproximado mínimo.

4. Para cada aresta $\{q_i, q_j\}$ da MST em $G'$, “traduzir” essa ligação de volta para o grafo original, usando o caminho correspondente entre os clusters $i$ e $j$.

5. Unindo:
   - as árvores locais $T_k$ de cada cluster, e
   - as ligações globais vindas da MST,

   obtemos uma árvore $T$ que conecta todos os terminais de todos os clusters.

### 3.4 Pseudocódigo (alto nível)

Pseudocódigo em alto nível para a heurística 2-níveis SPH+MST:

```text
Heuristica_2Niveis_SPH_MST(G = (V, E), custos c_e, clusters {R_1, ..., R_h})

  // Nível local: árvores dentro de cada cluster
  para k = 1 até h:
      T_k <- SPH_Local(G, R_k)
  fim

  // Escolher representantes (por simplicidade, 1 vértice por cluster)
  para k = 1 até h:
      q_k <- EscolherRepresentante(R_k)   // por exemplo, vértice de menor índice
  fim

  // Construir grafo dos clusters
  Q <- {q_1, ..., q_h}
  E' <- {}
  para todo par (i, j) com i < j:
      custo_ij <- MenorDistanciaEntreClusters(G, R_i, R_j)
      adiciona aresta {q_i, q_j} em E' com custo custo_ij
  fim

  // MST entre clusters
  T_global <- MST((Q, E'))

  // Traduzir para o grafo original e unir às árvores locais
  T <- Unir_Tks_com_Tglobal(G, {T_k}, T_global)

  retorna T
```