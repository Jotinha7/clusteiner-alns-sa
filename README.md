# CluSteiner ALNS-SA

Implementação de uma metaheurística **Adaptive Large Neighborhood Search (ALNS)** com critério de aceitação por **Simulated Annealing (SA)** para o **Clustered Steiner Tree Problem (CluSteiner)**.

## Visão geral

Este repositório reúne código, experimentos e documentação relacionados ao desenvolvimento de uma abordagem metaheurística para o problema **CluSteiner**. O foco principal é uma solução baseada em **ALNS-SA**.

## Tabela de comparação

O repositório inclui uma tabela com resultados de comparação entre algoritmos da literatura e a abordagem **ALNS** desenvolvida neste projeto.

Arquivo:

```text
tabela-comparacao-algoritmos.pdf
```

## Requisitos

- Python 3.11 ou superior

## Ambiente

Criação do ambiente virtual e instalação do projeto:

```bash
cd py
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cd ..
```

## Execução

Exemplo de comando para rodar uma instância a partir da raiz do repositório:

```bash
source py/.venv/bin/activate

python3 -m exp.run_alns_sa \
  --instance data/raw/NON_EUC_Type1_Small/10berlin52.txt \
  --time 1800 \
  --evals 25000 \
  --seed 0 \
  --sa_auto_t0 \
  --adaptive \
  --enable_reset \
  --reset_prob 0.02 \
  --reset_stag 3000 \
  --reset_cooldown 5000
```

Ajuste os caminhos e parâmetros de acordo com a organização local do projeto.

## Saídas experimentais

Dependendo da configuração, as execuções podem gerar:

- melhor custo encontrado;
- arquivos de solução;
- logs de iteração;
- resultados de validação;
- dados para tabelas.

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.