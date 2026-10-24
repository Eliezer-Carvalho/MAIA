<h1> <a href = https://github.com/Eliezer-Carvalho/MAIA/blob/master/DatasetBuilder/DatasetBuilder.ipynb> Construção dos Datasets </a> </h1>

O objetivo era que todos os datasets tivessem este formato:

```json
"role": "system", "content": "Alguma coisa",
"role": "user", "content": "Alguma coisa",
"role": "assistant", "content": "Alguma coisa"
```

Por isso foi necessário realizar um pré processamento, o código Python mais usado foi este:

```python
roles = set ()
keys = set ()

for ex in dataset["conversations"]:
    for chat in ex:
       
        roles.add (chat["role"]) 

        for y in chat.keys():
            keys.add (y)

print (f"Roles do Dataset: {roles}")
print (f"Keys do Dataset: {keys}")

def NORMALIZE_ROLES (dados):

    for exemplo in dados["conversations"]:
     
        if exemplo["role"] == "human":
            exemplo["role"] = "user"

        if exemplo["role"] == "gpt":
            exemplo["role"] = "assistant"

    return dados

dataset = dataset.map (NORMALIZE_ROLES)
```

Muitas das vezes também foi necessário concatenar datasets:

```python
datasetv1 = load_dataset ("amalia-llm/PT-Culture_Data", "v1")["train"]
datasetv2 = load_dataset ("amalia-llm/PT-Culture_Data", "v2")["train"]

dataset = concatenate_datasets ([datasetv1, datasetv2])
```

Por vezes também foi necessário remover algumas colunas:

```python
dataset = load_dataset ("amalia-llm/persona_instruction_following", "full")["pt"]

dataset = dataset.remove_columns (["persona", "constraints_used", "all_constraints_met"])
```

Ou renomear algumas:

```python
dataset = dataset.rename_column ("messages", "conversations")
```

No fim, juntou-se todos os datasets para o dataset final:

```python
dataset1 = load_dataset ("parquet", data_files = , split = "train")
dataset2 = load_dataset ("parquet", data_files = , split = "train")
dataset3 = load_dataset ("parquet", data_files = , split = "train")
dataset4 = load_dataset ("parquet", data_files = , split = "train")

dataset = concatenate_datasets ([dataset1, dataset2, dataset3, dataset4])
```

<h3> TL;DR </h3>

Alterações realizadas aos datasets originais com o objetivo de obter o formato pretendido.

| | Normalizar Nomes | Normalizar Roles | Normalizar Keys | Remover Colunas |
|----------------------------|:-----:|:----------------:| :---: | :---: |
| amalia-llm/PT-Culture_Data | :x: | :white_check_mark: | :x: | :white_check_mark: |
| amalia-llm/smol-rewrite-PT | :white_check_mark: | :x: | :x: | :x: |
| amalia-llm/amalia-PTradutor | :x: | :x: | :white_check_mark: | :x: |
| amalia-llm/persona_math | :x: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| amalia-llm/smol_summarize_pt | :x: | :x: | :x: | :x: |
| amalia-llm/persona_instruction_following | :x: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| amalia-llm/ptpt-linguistics-if | :x: | :x: | :white_check_mark: | :x: |
| amalia-llm/persona_general | :x: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| amalia-llm/smoltalk2_everyday_conv_pt | :x: | :white_check_mark: | :white_check_mark: | :x: |
| amalia-llm/wikipedia_conversations | :white_check_mark: | :x: | :white_check_mark: | :x: |
| amalia-llm/wmt24pp_xx_to_pt_10k | :x: | :x: | :white_check_mark: | :x: |



<h1> <a href = "https://github.com/Eliezer-Carvalho/MAIA/blob/master/DatasetBuilder/DatasetAnalysis.ipynb"> Análise dos Datasets </a> </h1>

Para cada dataset foi realizada uma análise de acordo com os seguintes parâmetros:

- Confirmação do formato do dataset
- Número de Exemplos
- Número de Palavras
- Número de Letras
- Número de Tokens do Conteúdo
- Número de Tokens de Treino

<b>Número de Tokens do Conteúdo vs Número de Tokes de Treino -></b> Foi realizada esta análise porque o conteúdo representa o comportamento que queremos ensinar ao modelo, enquanto os tokens de treino representam o número total de tokens que vão ser utilizados durante o processo de treino (com chat template incluído).

O código mais utilizado foi:

```python
WORDS = []
CHARS = []
TOKENS_CONTENT = []
TOKENS_TREINO = []

for exemplo in dataset["conversations"]:

    tok_treino = TOKENIZER.apply_chat_template (exemplo, add_generation_prompt = False, tokenize = True)
    TOKENS_TREINO.append (len (tok_treino))

    for x in exemplo:
       
        WORDS.append (len (x["content"].split()))
        
        tok_content = TOKENIZER (x["content"], add_special_tokens = False)
        TOKENS_CONTENT.append (len (tok_content["input_ids"]))
        
        CHARS.append (len (x["content"]))
        
print (f"Número de Exemplos: {dataset.num_rows}")
print (f"Número de Palavras: {sum(WORDS)}")
print (f"Número de Letras: {sum(CHARS)}")
print (f"Número de Tokens do Content: {sum(TOKENS_CONTENT)}")
print (f"Número de Tokens de Treino: {sum(TOKENS_TREINO)}")
```

<table>
    <thead>
        <th> cdcc </th>
    </thead>
    <tr>
        <th> cc ccdcd </th>
    </tr>
    <tr>
        <td> xsxsx </td>
    </tr>


</table>
