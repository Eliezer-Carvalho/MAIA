# %% [markdown]
# <h1> Libraries </h1>

# %%
from datasets import load_dataset
from transformers import AutoTokenizer

# %%
MODEL = r"Qwen/Qwen3-8B"

TOKENIZER = AutoTokenizer.from_pretrained (MODEL)

# %% [markdown]
# <hr>
# <h1> Dataset Analysis </h1>

# %% [markdown]
# <h4> amalia-llm/PT-Culture_Data </h4>

# %%
dataset = load_dataset (r"parquet", data_files = r"C:\Users\Admin\Desktop\MAIA\DatasetBuilder\Datasets\Dataset1.parquet", split = "train")

print (dataset[0])
print (dataset.features)

# %%
"""
Este código serve para confirmar se na fase de Construção do Dataset foi tudo bem realizado.
Confirma se o formato está no pretendido.
{'assistant', 'user', 'system'}
{'content', 'role'}
"""

roles = set ()
keys = set ()

for x in dataset["conversations"]:
    for exemplo in x:

        roles.add (exemplo["role"])
        #keys.add (list(exemplo.keys ()))
        for y in exemplo.keys():
        
            keys.add (y)
    
print (roles)
print (keys)

# %%
"""
x = "O que se comemora nas Festas de Alcoutim?"

tokens = TOKENIZER (x)
print (tokens["input_ids"])
print (len (tokens["input_ids"]))
"""

WORDS = []
CHARS = []
TOKENS = []

for exemplo in dataset["conversations"]:
    for x in exemplo:
        #print (x["content"])
        #print (x["content"].split()) # ['De', 'que', 'forma', 'o', 'provérbio', '“A', 'tenda', 'quer-se', 'com', 'quem', 'a', 'entenda”', 'reflete', 'a', 'sabedoria', 'popular', 'portuguesa?']

        WORDS.append (len (x["content"].split()))
        TOKENS.append (len (TOKENIZER (x["content"], add_special_tokens = False)["input_ids"]))
        CHARS.append (len (x["content"]))
        

        #print (len(letras))
        
        #WORDS.append (len (x["content"].split()))
        #TOKENS.append (len (TOKENIZER (x)["input_ids"]))

print (f"Número de Exemplos: {dataset.num_rows}")
print (f"Número de Palavras: {sum(WORDS)}")
print (f"Número de Letras: {sum(CHARS)}")
print (f"Número de Tokens: {sum(TOKENS)}")

# %% [markdown]
# <hr>
# <h4> amalia-llm/smol-rewrite-PT </h4>

# %%
dataset = load_dataset (r"parquet", data_files = r"C:\Users\Admin\Desktop\MAIA\DatasetBuilder\Datasets\Dataset2.parquet", split = "train")

print (dataset[0])
print (dataset.features)

# %%
"""
Este código serve para confirmar se na fase de Construção do Dataset foi tudo bem realizado.
Confirma se o formato está no pretendido.
{'assistant', 'user', 'system'}
{'content', 'role'}
"""

roles = set ()
keys = set ()

for x in dataset["conversations"]:
    for exemplo in x:

        roles.add (exemplo["role"])
        #keys.add (list(exemplo.keys ()))
        for y in exemplo.keys():
        
            keys.add (y)
    
print (roles)
print (keys)

# %%
"""
x = "O que se comemora nas Festas de Alcoutim?"

tokens = TOKENIZER (x)
print (tokens["input_ids"])
print (len (tokens["input_ids"]))
"""

WORDS = []
CHARS = []
TOKENS = []

for exemplo in dataset["conversations"]:
    for x in exemplo:
        #print (x["content"])
        #print (x["content"].split()) # ['De', 'que', 'forma', 'o', 'provérbio', '“A', 'tenda', 'quer-se', 'com', 'quem', 'a', 'entenda”', 'reflete', 'a', 'sabedoria', 'popular', 'portuguesa?']

        WORDS.append (len (x["content"].split()))
        TOKENS.append (len (TOKENIZER (x["content"], add_special_tokens = False)["input_ids"]))
        CHARS.append (len (x["content"]))
        

        #print (len(letras))
        
        #WORDS.append (len (x["content"].split()))
        #TOKENS.append (len (TOKENIZER (x)["input_ids"]))

print (f"Número de Exemplos: {dataset.num_rows}")
print (f"Número de Palavras: {sum(WORDS)}")
print (f"Número de Letras: {sum(CHARS)}")
print (f"Número de Tokens: {sum(TOKENS)}")

# %% [markdown]
# <hr>
# <h4> amalia-llm/amalia-PTradutor </h4>

# %%
dataset = load_dataset (r"parquet", data_files = r"C:\Users\Admin\Desktop\MAIA\DatasetBuilder\Datasets\Dataset3.parquet", split = "train")

print (dataset[0])
print (dataset.features)

# %%
"""
Este código serve para confirmar se na fase de Construção do Dataset foi tudo bem realizado.
Confirma se o formato está no pretendido.
{'assistant', 'user', 'system'}
{'content', 'role'}
"""

roles = set ()
keys = set ()

for x in dataset["conversations"]:
    for exemplo in x:

        roles.add (exemplo["role"])
        #keys.add (list(exemplo.keys ()))
        for y in exemplo.keys():
        
            keys.add (y)
    
print (roles)
print (keys)

# %%
"""
x = "O que se comemora nas Festas de Alcoutim?"

tokens = TOKENIZER (x)
print (tokens["input_ids"])
print (len (tokens["input_ids"]))
"""

WORDS = []
CHARS = []
TOKENS = []

for exemplo in dataset["conversations"]:
    for x in exemplo:
        #print (x["content"])
        #print (x["content"].split()) # ['De', 'que', 'forma', 'o', 'provérbio', '“A', 'tenda', 'quer-se', 'com', 'quem', 'a', 'entenda”', 'reflete', 'a', 'sabedoria', 'popular', 'portuguesa?']

        WORDS.append (len (x["content"].split()))
        TOKENS.append (len (TOKENIZER (x["content"], add_special_tokens = False)["input_ids"]))
        CHARS.append (len (x["content"]))
        

        #print (len(letras))
        
        #WORDS.append (len (x["content"].split()))
        #TOKENS.append (len (TOKENIZER (x)["input_ids"]))

print (f"Número de Exemplos: {dataset.num_rows}")
print (f"Número de Palavras: {sum(WORDS)}")
print (f"Número de Letras: {sum(CHARS)}")
print (f"Número de Tokens: {sum(TOKENS)}")

# %% [markdown]
# <hr>
# <h4> amalia-llm/persona_math </h4>

# %%
dataset = load_dataset (r"parquet", data_files = r"C:\Users\Admin\Desktop\MAIA\DatasetBuilder\Datasets\Dataset4.parquet", split = "train")

print (dataset[0])
print (dataset.features)

# %%
"""
Este código serve para confirmar se na fase de Construção do Dataset foi tudo bem realizado.
Confirma se o formato está no pretendido.
{'assistant', 'user', 'system'}
{'content', 'role'}
"""

roles = set ()
keys = set ()

for x in dataset["conversations"]:
    for exemplo in x:

        roles.add (exemplo["role"])
        #keys.add (list(exemplo.keys ()))
        for y in exemplo.keys():
        
            keys.add (y)
    
print (roles)
print (keys)

# %%
"""
x = "O que se comemora nas Festas de Alcoutim?"

tokens = TOKENIZER (x)
print (tokens["input_ids"])
print (len (tokens["input_ids"]))
"""

WORDS = []
CHARS = []
TOKENS = []

for exemplo in dataset["conversations"]:
    for x in exemplo:
        #print (x["content"])
        #print (x["content"].split()) # ['De', 'que', 'forma', 'o', 'provérbio', '“A', 'tenda', 'quer-se', 'com', 'quem', 'a', 'entenda”', 'reflete', 'a', 'sabedoria', 'popular', 'portuguesa?']

        WORDS.append (len (x["content"].split()))
        TOKENS.append (len (TOKENIZER (x["content"], add_special_tokens = False)["input_ids"]))
        CHARS.append (len (x["content"]))
        

        #print (len(letras))
        
        #WORDS.append (len (x["content"].split()))
        #TOKENS.append (len (TOKENIZER (x)["input_ids"]))

print (f"Número de Exemplos: {dataset.num_rows}")
print (f"Número de Palavras: {sum(WORDS)}")
print (f"Número de Letras: {sum(CHARS)}")
print (f"Número de Tokens: {sum(TOKENS)}")

# %% [markdown]
# <hr>
# <h4> amalia-llm/smol_summarize_pt </h4>

# %%
dataset = load_dataset (r"parquet", data_files = r"C:\Users\Admin\Desktop\MAIA\DatasetBuilder\Datasets\Dataset5.parquet", split = "train")

print (dataset[0])
print (dataset.features)

# %%
"""
Este código serve para confirmar se na fase de Construção do Dataset foi tudo bem realizado.
Confirma se o formato está no pretendido.
{'assistant', 'user', 'system'}
{'content', 'role'}
"""

roles = set ()
keys = set ()

for x in dataset["conversations"]:
    for exemplo in x:

        roles.add (exemplo["role"])
        #keys.add (list(exemplo.keys ()))
        for y in exemplo.keys():
        
            keys.add (y)
    
print (roles)
print (keys)

# %%
"""
x = "O que se comemora nas Festas de Alcoutim?"

tokens = TOKENIZER (x)
print (tokens["input_ids"])
print (len (tokens["input_ids"]))
"""

WORDS = []
CHARS = []
TOKENS = []

for exemplo in dataset["conversations"]:
    for x in exemplo:
        #print (x["content"])
        #print (x["content"].split()) # ['De', 'que', 'forma', 'o', 'provérbio', '“A', 'tenda', 'quer-se', 'com', 'quem', 'a', 'entenda”', 'reflete', 'a', 'sabedoria', 'popular', 'portuguesa?']

        WORDS.append (len (x["content"].split()))
        TOKENS.append (len (TOKENIZER (x["content"], add_special_tokens = False)["input_ids"]))
        CHARS.append (len (x["content"]))
        

        #print (len(letras))
        
        #WORDS.append (len (x["content"].split()))
        #TOKENS.append (len (TOKENIZER (x)["input_ids"]))

print (f"Número de Exemplos: {dataset.num_rows}")
print (f"Número de Palavras: {sum(WORDS)}")
print (f"Número de Letras: {sum(CHARS)}")
print (f"Número de Tokens: {sum(TOKENS)}")

# %% [markdown]
# <hr>
# <h4> amalia-llm/persona_instruction_following </h4>

# %%
dataset = load_dataset (r"parquet", data_files = r"C:\Users\Admin\Desktop\MAIA\DatasetBuilder\Datasets\Dataset6.parquet", split = "train")

print (dataset[0])
print (dataset.features)

# %%
"""
Este código serve para confirmar se na fase de Construção do Dataset foi tudo bem realizado.
Confirma se o formato está no pretendido.
{'assistant', 'user', 'system'}
{'content', 'role'}
"""

roles = set ()
keys = set ()

for x in dataset["conversations"]:
    for exemplo in x:

        roles.add (exemplo["role"])
        #keys.add (list(exemplo.keys ()))
        for y in exemplo.keys():
        
            keys.add (y)
    
print (roles)
print (keys)

# %%
"""
x = "O que se comemora nas Festas de Alcoutim?"

tokens = TOKENIZER (x)
print (tokens["input_ids"])
print (len (tokens["input_ids"]))
"""

WORDS = []
CHARS = []
TOKENS = []

for exemplo in dataset["conversations"]:
    for x in exemplo:
        #print (x["content"])
        #print (x["content"].split()) # ['De', 'que', 'forma', 'o', 'provérbio', '“A', 'tenda', 'quer-se', 'com', 'quem', 'a', 'entenda”', 'reflete', 'a', 'sabedoria', 'popular', 'portuguesa?']

        WORDS.append (len (x["content"].split()))
        TOKENS.append (len (TOKENIZER (x["content"], add_special_tokens = False)["input_ids"]))
        CHARS.append (len (x["content"]))
        

        #print (len(letras))
        
        #WORDS.append (len (x["content"].split()))
        #TOKENS.append (len (TOKENIZER (x)["input_ids"]))

print (f"Número de Exemplos: {dataset.num_rows}")
print (f"Número de Palavras: {sum(WORDS)}")
print (f"Número de Letras: {sum(CHARS)}")
print (f"Número de Tokens: {sum(TOKENS)}")

# %% [markdown]
# <hr>
# <h4> amalia-llm/ptpt-linguistics-if </h4>

# %%
dataset = load_dataset (r"parquet", data_files = r"C:\Users\Admin\Desktop\MAIA\DatasetBuilder\Datasets\Dataset7.parquet", split = "train")

print (dataset[0])
print (dataset.features)

# %%
"""
Este código serve para confirmar se na fase de Construção do Dataset foi tudo bem realizado.
Confirma se o formato está no pretendido.
{'assistant', 'user', 'system'}
{'content', 'role'}
"""

roles = set ()
keys = set ()

for x in dataset["conversations"]:
    for exemplo in x:

        roles.add (exemplo["role"])
        #keys.add (list(exemplo.keys ()))
        for y in exemplo.keys():
        
            keys.add (y)
    
print (roles)
print (keys)

# %%
"""
x = "O que se comemora nas Festas de Alcoutim?"

tokens = TOKENIZER (x)
print (tokens["input_ids"])
print (len (tokens["input_ids"]))
"""

WORDS = []
CHARS = []
TOKENS = []

for exemplo in dataset["conversations"]:
    for x in exemplo:
        #print (x["content"])
        #print (x["content"].split()) # ['De', 'que', 'forma', 'o', 'provérbio', '“A', 'tenda', 'quer-se', 'com', 'quem', 'a', 'entenda”', 'reflete', 'a', 'sabedoria', 'popular', 'portuguesa?']

        WORDS.append (len (x["content"].split()))
        TOKENS.append (len (TOKENIZER (x["content"], add_special_tokens = False)["input_ids"]))
        CHARS.append (len (x["content"]))
        

        #print (len(letras))
        
        #WORDS.append (len (x["content"].split()))
        #TOKENS.append (len (TOKENIZER (x)["input_ids"]))

print (f"Número de Exemplos: {dataset.num_rows}")
print (f"Número de Palavras: {sum(WORDS)}")
print (f"Número de Letras: {sum(CHARS)}")
print (f"Número de Tokens: {sum(TOKENS)}")

# %% [markdown]
# <hr>
# <h4> amalia-llm/persona_general </h4>

# %%
dataset = load_dataset (r"parquet", data_files = r"C:\Users\Admin\Desktop\MAIA\DatasetBuilder\Datasets\Dataset8.parquet", split = "train")

print (dataset[0])
print (dataset.features)

# %%
"""
Este código serve para confirmar se na fase de Construção do Dataset foi tudo bem realizado.
Confirma se o formato está no pretendido.
{'assistant', 'user', 'system'}
{'content', 'role'}
"""

roles = set ()
keys = set ()

for x in dataset["conversations"]:
    for exemplo in x:

        roles.add (exemplo["role"])
        #keys.add (list(exemplo.keys ()))
        for y in exemplo.keys():
        
            keys.add (y)
    
print (roles)
print (keys)

# %%
"""
x = "O que se comemora nas Festas de Alcoutim?"

tokens = TOKENIZER (x)
print (tokens["input_ids"])
print (len (tokens["input_ids"]))
"""

WORDS = []
CHARS = []
TOKENS = []

for exemplo in dataset["conversations"]:
    for x in exemplo:
        #print (x["content"])
        #print (x["content"].split()) # ['De', 'que', 'forma', 'o', 'provérbio', '“A', 'tenda', 'quer-se', 'com', 'quem', 'a', 'entenda”', 'reflete', 'a', 'sabedoria', 'popular', 'portuguesa?']

        WORDS.append (len (x["content"].split()))
        TOKENS.append (len (TOKENIZER (x["content"], add_special_tokens = False)["input_ids"]))
        CHARS.append (len (x["content"]))
        

        #print (len(letras))
        
        #WORDS.append (len (x["content"].split()))
        #TOKENS.append (len (TOKENIZER (x)["input_ids"]))

print (f"Número de Exemplos: {dataset.num_rows}")
print (f"Número de Palavras: {sum(WORDS)}")
print (f"Número de Letras: {sum(CHARS)}")
print (f"Número de Tokens: {sum(TOKENS)}")

# %% [markdown]
# <hr>
# <h4> amalia-llm/smoltalk2_everyday_conv_pt </h4>

# %%
dataset = load_dataset (r"parquet", data_files = r"C:\Users\Admin\Desktop\MAIA\DatasetBuilder\Datasets\Dataset9.parquet", split = "train")

print (dataset[0])
print (dataset.features)

# %%
"""
Este código serve para confirmar se na fase de Construção do Dataset foi tudo bem realizado.
Confirma se o formato está no pretendido.
{'assistant', 'user', 'system'}
{'content', 'role'}
"""

roles = set ()
keys = set ()

for x in dataset["conversations"]:
    for exemplo in x:

        roles.add (exemplo["role"])
        #keys.add (list(exemplo.keys ()))
        for y in exemplo.keys():
        
            keys.add (y)
    
print (roles)
print (keys)

# %%
"""
x = "O que se comemora nas Festas de Alcoutim?"

tokens = TOKENIZER (x)
print (tokens["input_ids"])
print (len (tokens["input_ids"]))
"""

WORDS = []
CHARS = []
TOKENS = []

for exemplo in dataset["conversations"]:
    for x in exemplo:
        #print (x["content"])
        #print (x["content"].split()) # ['De', 'que', 'forma', 'o', 'provérbio', '“A', 'tenda', 'quer-se', 'com', 'quem', 'a', 'entenda”', 'reflete', 'a', 'sabedoria', 'popular', 'portuguesa?']

        WORDS.append (len (x["content"].split()))
        TOKENS.append (len (TOKENIZER (x["content"], add_special_tokens = False)["input_ids"]))
        CHARS.append (len (x["content"]))
        

        #print (len(letras))
        
        #WORDS.append (len (x["content"].split()))
        #TOKENS.append (len (TOKENIZER (x)["input_ids"]))

print (f"Número de Exemplos: {dataset.num_rows}")
print (f"Número de Palavras: {sum(WORDS)}")
print (f"Número de Letras: {sum(CHARS)}")
print (f"Número de Tokens: {sum(TOKENS)}")

# %% [markdown]
# <hr>
# <h4> amalia-llm/wikipedia_conversations </h4>

# %%
dataset = load_dataset (r"parquet", data_files = r"C:\Users\Admin\Desktop\MAIA\DatasetBuilder\Datasets\Dataset10.parquet", split = "train")

print (dataset[0])
print (dataset.features)

# %%
"""
Este código serve para confirmar se na fase de Construção do Dataset foi tudo bem realizado.
Confirma se o formato está no pretendido.
{'assistant', 'user', 'system'}
{'content', 'role'}
"""

roles = set ()
keys = set ()

for x in dataset["conversations"]:
    for exemplo in x:

        roles.add (exemplo["role"])
        #keys.add (list(exemplo.keys ()))
        for y in exemplo.keys():
        
            keys.add (y)
    
print (roles)
print (keys)

# %%
"""
x = "O que se comemora nas Festas de Alcoutim?"

tokens = TOKENIZER (x)
print (tokens["input_ids"])
print (len (tokens["input_ids"]))
"""

WORDS = []
CHARS = []
TOKENS = []

for exemplo in dataset["conversations"]:
    for x in exemplo:
        #print (x["content"])
        #print (x["content"].split()) # ['De', 'que', 'forma', 'o', 'provérbio', '“A', 'tenda', 'quer-se', 'com', 'quem', 'a', 'entenda”', 'reflete', 'a', 'sabedoria', 'popular', 'portuguesa?']

        WORDS.append (len (x["content"].split()))
        TOKENS.append (len (TOKENIZER (x["content"], add_special_tokens = False)["input_ids"]))
        CHARS.append (len (x["content"]))
        

        #print (len(letras))
        
        #WORDS.append (len (x["content"].split()))
        #TOKENS.append (len (TOKENIZER (x)["input_ids"]))

print (f"Número de Exemplos: {dataset.num_rows}")
print (f"Número de Palavras: {sum(WORDS)}")
print (f"Número de Letras: {sum(CHARS)}")
print (f"Número de Tokens: {sum(TOKENS)}")

# %% [markdown]
# <hr>
# <h4> amalia-llm/wmt24pp_xx_to_pt_10k </h4>

# %%
dataset = load_dataset (r"parquet", data_files = r"C:\Users\Admin\Desktop\MAIA\DatasetBuilder\Datasets\Dataset11.parquet", split = "train")

print (dataset[0])
print (dataset.features)

# %%
"""
Este código serve para confirmar se na fase de Construção do Dataset foi tudo bem realizado.
Confirma se o formato está no pretendido.
{'assistant', 'user', 'system'}
{'content', 'role'}
"""

roles = set ()
keys = set ()

for x in dataset["conversations"]:
    for exemplo in x:

        roles.add (exemplo["role"])
        #keys.add (list(exemplo.keys ()))
        for y in exemplo.keys():
        
            keys.add (y)
    
print (roles)
print (keys)

# %%
"""
x = "O que se comemora nas Festas de Alcoutim?"

tokens = TOKENIZER (x)
print (tokens["input_ids"])
print (len (tokens["input_ids"]))
"""

WORDS = []
CHARS = []
TOKENS = []

for exemplo in dataset["conversations"]:
    for x in exemplo:
        #print (x["content"])
        #print (x["content"].split()) # ['De', 'que', 'forma', 'o', 'provérbio', '“A', 'tenda', 'quer-se', 'com', 'quem', 'a', 'entenda”', 'reflete', 'a', 'sabedoria', 'popular', 'portuguesa?']

        WORDS.append (len (x["content"].split()))
        TOKENS.append (len (TOKENIZER (x["content"], add_special_tokens = False)["input_ids"]))
        CHARS.append (len (x["content"]))
        

        #print (len(letras))
        
        #WORDS.append (len (x["content"].split()))
        #TOKENS.append (len (TOKENIZER (x)["input_ids"]))

print (f"Número de Exemplos: {dataset.num_rows}")
print (f"Número de Palavras: {sum(WORDS)}")
print (f"Número de Letras: {sum(CHARS)}")
print (f"Número de Tokens: {sum(TOKENS)}")

# %% [markdown]
# <hr>
# <h1> Analysis Dataset </h1>

# %%
dataset = load_dataset ("parquet", data_files = r"C:\Users\Admin\Desktop\MAIA\DatasetBuilder\Datasets\Dataset.parquet", split = "train")

"""
x = "O que se comemora nas Festas de Alcoutim?"

tokens = TOKENIZER (x)
print (tokens["input_ids"])
print (len (tokens["input_ids"]))
"""

WORDS = []
CHARS = []
TOKENS = []

for exemplo in dataset["conversations"]:
    for x in exemplo:
        #print (x["content"])
        #print (x["content"].split()) # ['De', 'que', 'forma', 'o', 'provérbio', '“A', 'tenda', 'quer-se', 'com', 'quem', 'a', 'entenda”', 'reflete', 'a', 'sabedoria', 'popular', 'portuguesa?']

        WORDS.append (len (x["content"].split()))
        TOKENS.append (len (TOKENIZER (x["content"], add_special_tokens = False)["input_ids"]))
        CHARS.append (len (x["content"]))
        

        #print (len(letras))
        
        #WORDS.append (len (x["content"].split()))
        #TOKENS.append (len (TOKENIZER (x)["input_ids"]))

print (f"Número de Exemplos: {dataset.num_rows}")
print (f"Número de Palavras: {sum(WORDS)}")
print (f"Número de Letras: {sum(CHARS)}")
print (f"Número de Tokens: {sum(TOKENS)}")


