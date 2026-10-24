<h1> <a href = https://github.com/Eliezer-Carvalho/MAIA/blob/master/dataset/DatasetBuilder.ipynb> Construção dos Datasets </a> </h1>

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



<h1> <a href = "https://github.com/Eliezer-Carvalho/MAIA/blob/master/DatasetBuilder/DatasetAnalysis.ipynbhttps://github.com/Eliezer-Carvalho/MAIA/blob/master/dataset/DatasetAnalysis.ipynb"> Análise dos Datasets </a> </h1>

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

<h3> TL;DR  </h3>

| | Exemplos | Palavras | Letras | Tokens Conteúdo | Tokens Treino |
| ------------------------------------------ | :-----: | :-----: | :-----: | :-----: | :-----: | 
| amalia-llm/PT-Culture_Data                 |216 832|21 290 738|129 407 253|36 852 495|40 341 173|         
| amalia-llm/smol-rewrite-PT                 |53 342         |14 033 826         |92 453 779         |24 257 416         |25 270 914         |         
| amalia-llm/amalia-PTradutor                |50 000         |17 299 112         |103 884 811         |26 283 002         |27 107 702         |         
| amalia-llm/persona_math                    |63 731         |48 518 265         |290 879 157         |96 606 701         |97 498 935         |         
| amalia-llm/smol_summarize_pt               |96 356         |39 030 934         |241 111 800         |65 552 775         |67 143 784         |         
| amalia-llm/persona_instruction_following   |27 781         |22 886 136         |157 474 187         |43 630 962         |44 019 896         |         
| amalia-llm/ptpt-linguistics-if             |601         |35 629         |213 342         |64 680         |73 094         |         
| amalia-llm/persona_general                 |156 397         |117 000 905         |811 238 762         |218 740 646         |220 930 203         |         
| amalia-llm/smoltalk2_everyday_conv_pt      |2255         |548 681         |3 248 484         |902 809         |1 012 460         |         
| amalia-llm/wikipedia_conversations         |97 553         |52 722 546         |321 718 477         |89 695 968         |94 610 420         |         
| amalia-llm/wmt24pp_xx_to_pt_10k            |10 000         |752 131         |4 698 926         |1 823 442         |1 988 847         |         
| <b> Dataset Final </b> | <b> 774 848 </b> | <b>334 118 903</b> | <b>2 156 328 978</b> | <b>604 410 896</b> | <b>619 997 428</b> |

<i> Como é óbvio, os dados do Dataset Final correspondem à soma das labels dos datasets originais. | Dados de 24/10/2026 </i>