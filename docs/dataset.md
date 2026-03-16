# Dataset – Clustered Steiner Tree (CluSteiner)

Este documento descreve o dataset usado neste TCC, extraído do repositório
**CluSteiner-Dataset**, publicado pelos autores do artigo:

> *An online transfer learning based multifactorial evolutionary algorithm for solving the clustered Steiner tree problem (2024).*

O objetivo é explicar de forma clara:
- a origem das instâncias;  
- como o dataset foi adaptado;  
- o formato de cada arquivo;  
- e como essas informações serão usadas no projeto.

---

## 1. Fonte do Dataset

O dataset original é baseado no conjunto **MOM-LIB**, que contém instâncias
do problema **Clustered Traveling Salesman Problem (CluTSP)**.

Os autores do artigo adaptam essas instâncias para o problema  
**Clustered Steiner Tree (CluSteiner)** com o seguinte processo:

- cada instância possui um conjunto de **vértices terminais obrigatórios** (R);  
- R é dividido em **clusters** (R₁, R₂, …, Rₕ);  
- todos os vértices em R **devem aparecer na solução**;  
- vértices fora de R são **Steiner opcionais**.

O repositório contém instâncias:

- **Euclidianas** (distância EUC_2D);
- **Não-Euclidianas** (distâncias arbitrárias).

---

## 2. Organização no Projeto

- As pastas que começam com `EUC_` contêm instâncias **euclidianas**.  
- As que começam com `NON_EUC_` contêm instâncias **não-euclidianas**.  
- Em cada pasta, **cada arquivo `.txt` é uma instância** do problema.

---

## 3. Estrutura de uma Instância

Cada arquivo segue um formato inspirado na TSPLIB/GTSP.

Exemplo (instância `10berlin52.txt`):

```
Name : 10berlin52
DIMENSION : 52
GTSP_SETS : 10
NODE_COORD_SECTION
1 565 575
2 25 185
3 345 750
4 945 685
...
52 1740 245
GTSP_SET_SECTION
1 1 -1
2 13 28 52 -1
3 2 -1
4 16 20 -1
5 4 40 46 -1
6 9 -1
7 33 -1
8 8 45 -1
9 11 12 -1
10 3 -1
```

### 3.1. Cabeçalho

- **Name**: nome da instância (ex.: `10berlin52`);  
- **DIMENSION**: número total de vértices |V|;  
- **GTSP_SETS**: número de clusters h.

### 3.2. NODE_COORD_SECTION

Lista (id, x, y) de **todos** os vértices do grafo.

Exemplo:

- `1 565 575` → vértice 1 tem coordenadas (565, 575);  
- `2 25 185` → vértice 2 tem coordenadas (25, 185).

Todos os vértices de 1 até DIMENSION aparecem aqui.

### 3.3. GTSP_SET_SECTION

Define os **vértices terminais obrigatórios**, agrupados em clusters.

Interpretando o exemplo:

- Cluster 1: {1}  
- Cluster 2: {13, 28, 52}  
- Cluster 3: {2}  
- Cluster 4: {16, 20}  
- Cluster 5: {4, 40, 46}  
- Cluster 6: {9}  
- Cluster 7: {33}  
- Cluster 8: {8, 45}  
- Cluster 9: {11, 12}  
- Cluster 10: {3}

O conjunto total de terminais é:

R = {1, 2, 3, 4, 8, 9, 11, 12, 13, 16, 20, 28, 33, 40, 45, 46, 52}.

---

## 4. Terminais e Vértices Steiner

Na formulação do problema, a árvore de Steiner clusterizada T deve:

- ser uma árvore de Steiner em G;  
- conter **todos os vértices terminais** listados nos clusters.

Logo:

- todos os vértices listados na seção GTSP_SET_SECTION são **terminais obrigatórios**;  
- qualquer vértice não listado é um **vértice Steiner opcional**.

No exemplo:

- DIMENSION = 52 vértices no grafo;  
- |R| = 17 terminais;  
- vértices Steiner = 52 − 17 = 35.

---

## 5. Tipos de Instância

O dataset possui seis tipos (1 a 6).  
Utilizamos os **tipos 1, 5 e 6**, em versões **Small** e **Large**, com variantes euclidianas e não-euclidianas.

### Tipo 1  
Clusters mais definidos.

### Tipo 5  
Clusters geométricos.

### Tipo 6  
Clusters irregulares.

Cada tipo possui versões:

- **Small** → instâncias menores (~50–100 nós)  
- **Large** → instâncias maiores (~250–450 nós)

---

## 6. Resumo para Implementação

Para cada arquivo `.txt`, o parser deve:

1. Ler o cabeçalho (Name, DIMENSION, GTSP_SETS ou NUMBER_OF_CLUSTERS).  
2. Ler NODE_COORD_SECTION (coordenadas dos vértices).  
3. Ler GTSP_SET_SECTION (ou CLUSTER_SECTION nas não-euclidianas).  
4. Calcular:
   - número total de vértices;  
   - número de terminais;  
   - vértices Steiner opcionais;  
   - clusters e seus tamanhos.  
5. Gerar o arquivo CSV em `data/processed/instances.csv`.


