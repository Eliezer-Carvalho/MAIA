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

Nesta etapa estudámos o Dataset Tokenizado a partir da Distribuição da Sequence Length. <br>
Anteriormente, a Tokenização foi realizada sem Truncation nem Padding, pois nesta fase apenas procuramos estudar a distribuição da dataset.

















Datasets Fugazi criados com os valores das distribuições calculadas do Dataset Original. Em formato JSON apenas no GitHub.

<h1> FugaziDatasetBuilder </h1>

Código de criação e validação dos datasets.

<h1> SequenceLengthDistribution </h1>

Análise da Sequence Length Distribution

<h1> Tokenizer </h1>

Código para realizar a Tokenização do Dataset Original.

```mermaid
    flowchart TB
    n1["Dataset Final"] --> n2["Tokenização"]
    n2 --> n3["Análise Destribuição Sequence Length"]
    n3 --> n4["Média"] & n5["Mediana"] & n6["P75"] & n7["P90"] & n8["P95"] & n9["P99"]
    n4 --> n11["Fugazi Dataset — Fixed Sequence Length Derived from the Original Sequence-Length Distribution"]
    n5 --> n11
    n6 --> n11
    n7 --> n11
    n8 --> n11
    n9 --> n11