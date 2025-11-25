# Few-Shot vs Zero-Shot Learning

Comparacao de tecnicas de few-shot e zero-shot learning para classificacao de intencoes usando LLMs.

## Dataset

Este projeto usa o **Banking77**, um dataset de classificacao de intencoes com 77 categorias de perguntas bancarias.

- ~10.000 exemplos de treino
- ~3.000 exemplos de teste
- 77 intencoes diferentes (ex: `card_arrival`, `activate_my_card`, `request_refund`)

### Download dos dados

```bash
mkdir -p data
cd data

# Train set
curl -L "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv" -o banking77_train.csv

# Test set
curl -L "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv" -o banking77_test.csv
```

## Configuracao

1. Crie um arquivo `.env` com sua chave da OpenAI:

```
OPENAI_API_KEY=sua_chave_aqui
```

2. Instale as dependencias:

```bash
uv sync
```

## Execucao

```bash
uv run python main.py
```

## Referencias

- [Banking77 Dataset](https://github.com/PolyAI-LDN/task-specific-datasets)
- [Paper Original](https://arxiv.org/abs/2003.04807)
