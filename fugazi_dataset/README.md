<h1> <a href = > Tokenização do Dataset </a> </h1>

Antes de tudo, tivemos que converter o Dataset para o cenário real de treino. <b>Onde o mesmo está Tokenizado.</b> <br>
Não foi preciso usar <i>streaming </i> para realizar a Tokenização. <br>

```python
from transformers import AutoTokenizer
from datasets import load_dataset

MODEL = "Qwen/Qwen3-8B"
TOKENIZER = AutoTokenizer.from_pretrained (MODEL)

dataset = load_dataset ("parquet", data_files = r"C:\Users\User\Desktop\MAIA\dataset\Datasets\Dataset.parquet", split = "train")

def TOKENIZAÇÃO (dados):
  
    EXEMPLO = TOKENIZER.apply_chat_template (dados["conversations"], tokenize = False, add_generation_prompt = False, enable_thinking = False) 
    TOKENS = TOKENIZER (EXEMPLO, add_special_tokens = False, padding = False, truncation = False)

    return TOKENS

dataset_tokenizado = dataset.map (TOKENIZAÇÃO, remove_columns = "conversations")
dataset_tokenizado.to_parquet ("DatasetTokenizado.parquet")
```

<h1> <a href = > Sequence Length Distribution</a> </h1>

Nesta etapa estudámos o Dataset Tokenizado a partir da <b>Distribuição da Sequence Length</b>. <br>
Anteriormente, a Tokenização foi realizada sem Truncation nem Padding, pois nesta fase apenas procuramos estudar a distribuição do dataset.


| | SLD Minimo | SLD Máximo | SLD Média | SLD P50/Mediana | SLD P75 | SLD P90 | SLD P95 | SLD P99
|-| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | 
| Dataset Tokenizado S/Truncation & Padding | 26 | 5128 | 800 | 727 | 1352 | 1594 | 1716 | 2179       

<i> Valores em Tokens | SLD -> Sequence Length Distribution </i>








