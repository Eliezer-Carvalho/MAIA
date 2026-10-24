<h1> <a href = https://github.com/Eliezer-Carvalho/MAIA/blob/master/fugazi_dataset/Dataset/tokenizer.py> Tokenização do Dataset </a> </h1>

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

<h1> <a href = https://github.com/Eliezer-Carvalho/MAIA/blob/master/fugazi_dataset/Dataset/sequence_len_distribution.ipynb > Sequence Length Distribution</a> </h1>

Nesta etapa estudámos o Dataset Tokenizado a partir da <b>Distribuição da Sequence Length</b>. <br>
Anteriormente, a Tokenização foi realizada sem Truncation nem Padding, pois nesta fase apenas procuramos estudar a distribuição do dataset.


| | SLD Minimo | SLD Máximo | SLD Média | SLD P50/Mediana | SLD P75 | SLD P90 | SLD P95 | SLD P99
|-| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | 
| Dataset Tokenizado S/Truncation & Padding | 26 | 5128 | 800 | 727 | 1352 | 1594 | 1716 | 2179       

<i> Valores em Tokens | SLD = <b> Sequence Length Distribution </b> </i>

<h3> Interpretação da Tabela </h3>
Neste dataset o menor exemplo, tem <b> 26 tokens, </b>o maior tem <b> 5128 tokens</b> e a média por exemplo é de <b> 800 tokens</b>. <br>
No dataset <b>95% dos exemplos têm 1716 tokens ou menos</b> de comprimento.


<h1> <a href = > Construção dos Datasets Fugazi </a> </h1>

Após a análise da Sequence Length Distribution, construímos os Datasets Fugazi usando o valor da:

- Média
- P50/Mediana
- P75
- P90
- P95
- P99

```python
dados = []

for x in dataset:
 
    if len (x["input_ids"]) == _config["Dataset Tokenizado"]["P99"]: 

        print (f"Tokens -> {x['input_ids']}") 
        print ("---" * 50)

        print (f"Número de Tokens -> {len (x['input_ids'])}") 
        print ("---" * 50)

        print (f"Exemplo Real -> {TOKENIZER.decode (x['input_ids'], add_special_tokens = False)}") 

        for i in range (100):

            dados.append (x) 

        break


repeat = set ()
for i in dados:
    #print (i)
    repeat.add (tuple (i["input_ids"])) 

print ("---" * 50) 
print (f"Número de Repetidos -> {len (repeat) - 1}") 

dataset_p99 = Dataset.from_list (dados)   
dataset_p99.to_parquet ("Dataset_p99.parquet")

dataset_ = load_dataset ("parquet", data_files = "Dataset_p99.parquet", split = "train")
print (dataset_.features)
print (dataset_ [0])
```





