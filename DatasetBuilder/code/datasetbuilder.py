# %% [markdown]
# <h1> Libraries </h1>

# %%
from datasets import load_dataset
from datasets import concatenate_datasets

# %% [markdown]
# <hr>
# <h1> Dataset Builder </h1>

# %% [markdown]
# <h4> amalia-llm/PT-Culture_Data </h4>

# %%
datasetv1 = load_dataset ("amalia-llm/PT-Culture_Data", "v1")["train"]

display (datasetv1.features)
display (datasetv1[0])
print (type(datasetv1))
print ("---" *50)

datasetv2 = load_dataset ("amalia-llm/PT-Culture_Data", "v2")["train"]

display (datasetv2.features)
display (datasetv2[0])
print (type(datasetv2))
print ("---" *50)

dataset = concatenate_datasets ([datasetv1, datasetv2])

display (dataset.features)
display (dataset[0])
print (type(dataset))

# %%
dataset = dataset.remove_columns (["category", "_task_type", "_seed_id"])

print (dataset.features)

# %%
roles = set ()
keys = set ()

for ex in dataset["conversations"]:
    for chat in ex:
        #print (chat["role"])
        roles.add (chat["role"]) # {'gpt', 'human'}

        for y in chat.keys():
            keys.add (y)

print (f"Roles do Dataset: {roles}")
print (f"Keys do Dataset: {keys}")

# %%
"""
for exemplo in dataset["conversations"][:5]:
    for y in exemplo:
        #y["role"] = "user"
        #y["content"] = y.pop ("value")
        if y["role"] == "human":
            y["role"] = "user"

        if y["role"] == "gpt":
            y["role"] = "assistant"
"""


def NORMALIZE_ROLES (dados):

    for exemplo in dados["conversations"]:
        #y["role"] = "user"
        #y["content"] = y.pop ("value")
        if exemplo["role"] == "human":
            exemplo["role"] = "user"

        if exemplo["role"] == "gpt":
            exemplo["role"] = "assistant"

    return dados

dataset = dataset.map (NORMALIZE_ROLES)

print (dataset[0])

dataset.to_parquet ("Dataset1.parquet")

# %% [markdown]
# <hr>
# <h4> amalia-llm/smol-rewrite-PT </h4>

# %%
dataset = load_dataset ("amalia-llm/smol-rewrite-PT", "base")["train"]

display (dataset.features)
display (dataset[0])
print (type(dataset))


# %%
dataset = dataset.rename_column ("messages", "conversations")

# %%
roles = set ()
keys = set ()

for ex in dataset["conversations"]:
    for chat in ex:
        #print (chat["role"])
        roles.add (chat["role"]) # {'gpt', 'human'}

        for y in chat.keys():
            keys.add (y)

print (f"Roles do Dataset: {roles}")
print (f"Keys do Dataset: {keys}")

dataset.to_parquet ("Dataset2.parquet") # {'assistant', 'user', 'system'}

# %% [markdown]
# <hr>
# <h4> amalia-llm/amalia-PTradutor </h4>

# %%
dataset = load_dataset ("amalia-llm/amalia-PTradutor", "base")["train"]

display (dataset.features)
display (dataset[0])
print (type(dataset))

# %%
roles = set ()
keys = set ()

for ex in dataset["conversations"]:
    for chat in ex:
        #print (chat["role"])
        roles.add (chat["from"]) # {'gpt', 'human'}

        for y in chat.keys():
            keys.add (y)

print (f"Roles do Dataset: {roles}")
print (f"Keys do Dataset: {keys}")


# %%
"""
for x in dataset["conversations"][:1]:
    for exemplo in x:

        exemplo["role"] = exemplo.pop ("from")    
        exemplo["content"] = exemplo.pop ("value")
        print (exemplo)
"""

def NORMALIZAR_ROLE_CONTENT (dados):

    for x in dados["conversations"]:

        x["role"] = x.pop ("from")
        x["content"] = x.pop ("value")

    return dados

dataset = dataset.map (NORMALIZAR_ROLE_CONTENT)

print (dataset[0])

dataset.to_parquet ("Dataset3.parquet")

# %% [markdown]
# <hr>
# <h4> amalia-llm/persona_math </h4>

# %%
dataset = load_dataset ("amalia-llm/persona_math")["default"]

display (dataset.features)
display (dataset[0])
print (type(dataset))

dataset = dataset.remove_columns (["input persona", "quality_score"])

display (dataset.features)

# %%
roles = set ()
keys = set ()

for ex in dataset["conversations"]:
    for chat in ex:
        #print (chat["role"])
        roles.add (chat["from"]) # {'gpt', 'human'}

        for y in chat.keys():
            keys.add (y)

print (f"Roles do Dataset: {roles}")
print (f"Keys do Dataset: {keys}")


# %%
def FULL_NORMALIZAR (dados):

    for x in dados["conversations"]:

        x["role"] = x.pop("from")
        x["content"] = x.pop("value")

        if x["role"] == "human":
            x["role"] = "user"

        if x["role"] == "gpt":
            x["role"] = "assistant"

    return dados

dataset = dataset.map (FULL_NORMALIZAR)

print (dataset[0])
print (dataset.features)

dataset.to_parquet ("Dataset4.parquet")

# %% [markdown]
# <hr>
# <h4> amalia-llm/smol_summarize_pt </h4>

# %%
dataset = load_dataset ("amalia-llm/smol_summarize_pt")["train"]

display (dataset.features)
display (dataset[0])
print (type(dataset))

# %%
roles = set ()
keys = set ()

for ex in dataset["conversations"]:
    for chat in ex:
        #print (chat["role"])
        roles.add (chat["role"]) # {'gpt', 'human'}

        for y in chat.keys():
            keys.add (y)

print (f"Roles do Dataset: {roles}")
print (f"Keys do Dataset: {keys}")


# %%
dataset.to_parquet ("Dataset5.parquet")

# %% [markdown]
# <hr>
# <h4> amalia-llm/persona_instruction_following </h4>

# %%
dataset = load_dataset ("amalia-llm/persona_instruction_following", "full")["pt"]

display (dataset.features)
display (dataset[27779])
print (type(dataset))
print (dataset)

dataset = dataset.remove_columns (["persona", "constraints_used", "all_constraints_met"])

display (dataset.features)
display (dataset[5])
print (type(dataset))
print (dataset)

# %%
roles = set ()
keys = set ()

for ex in dataset["conversations"]:
    for chat in ex:
        #print (chat["role"])
        roles.add (chat["from"]) # {'gpt', 'human'}

        for y in chat.keys():
            keys.add (y)

print (f"Roles do Dataset: {roles}")
print (f"Keys do Dataset: {keys}")


# %%
def FULL_NORMALIZAR (dados):

    for x in dados["conversations"]:

        x["role"] = x.pop("from")
        x["content"] = x.pop("value")

        if x["role"] == "gpt":
            x["role"] = "assistant"

    return dados

dataset = dataset.map (FULL_NORMALIZAR)

print (dataset[0])
print (dataset.features)

dataset.to_parquet ("Dataset6.parquet")

# %% [markdown]
# <hr>
# <h4> amalia-llm/ptpt-linguistics-if </h4>

# %%
dataset = load_dataset ("amalia-llm/ptpt-linguistics-if")["train"]

display (dataset.features)
display (dataset[0])
print (type(dataset))

# %%
roles = set ()
keys = set ()

for ex in dataset["conversations"]:
    for chat in ex:
        #print (chat["role"])
        roles.add (chat["from"]) # {'gpt', 'human'}

        for y in chat.keys():
            keys.add (y)

print (f"Roles do Dataset: {roles}")
print (f"Keys do Dataset: {keys}")


# %%
def FULL_NORMALIZAR (dados):

    for x in dados["conversations"]:

        x["role"] = x.pop("from")
        x["content"] = x.pop("value")

    return dados

dataset = dataset.map (FULL_NORMALIZAR)

print (dataset[0])
print (dataset.features)

dataset.to_parquet ("Dataset7.parquet")

# %% [markdown]
# <hr>
# <h4> amalia-llm/persona_general </h4>

# %%
dataset = load_dataset ("amalia-llm/persona_general")["train"]

display (dataset.features)
display (dataset[0])
print (type(dataset))

dataset = dataset.remove_columns (["input persona", "quality_score"])

display (dataset.features)
display (dataset[0])
print (type(dataset))

# %%
roles = set ()
keys = set ()

for ex in dataset["conversations"]:
    for chat in ex:
        #print (chat["role"])
        roles.add (chat["from"]) # {'gpt', 'human'}

        for y in chat.keys():
            keys.add (y)

print (f"Roles do Dataset: {roles}")
print (f"Keys do Dataset: {keys}")


# %%
def FULL_NORMALIZAR (dados):

    for x in dados["conversations"]:

        x["role"] = x.pop("from")
        x["content"] = x.pop("value")

        if x["role"] == "human":
            x["role"] = "user"

        if x["role"] == "gpt":
            x["role"] = "assistant"

    return dados

dataset = dataset.map (FULL_NORMALIZAR)

print (dataset[0])
print (dataset.features)

dataset.to_parquet ("Dataset8.parquet")

# %% [markdown]
# <hr>
# <h4> amalia-llm/smoltalk2_everyday_conv_pt </h4>

# %%
dataset = load_dataset ("amalia-llm/smoltalk2_everyday_conv_pt")["train"]

display (dataset.features)
display (dataset[0])
print (dataset)
print ("---" *50)

dataset = dataset.remove_columns (["style", "meta"])

display (dataset.features)
display (dataset[0])
print (dataset)

# %%
dataset = dataset.rename_column ("messages", "conversations")

# %%
roles = set ()
keys = set ()

for ex in dataset["conversations"]:
    for chat in ex:
        #print (chat["role"])
        roles.add (chat["role"]) # {'gpt', 'human'}

        for y in chat.keys():
            keys.add (y)

print (f"Roles do Dataset: {roles}")
print (f"Keys do Dataset: {keys}")

dataset.to_parquet ("Dataset9.parquet")

# %% [markdown]
# <hr>
# <h4> amalia-llm/wikipedia_conversations </h4>

# %%
dataset = load_dataset ("amalia-llm/wikipedia_conversations")["train"]

display (dataset.features)
display (dataset[0])
print (dataset)
print ("---" *50)

# %%
dataset = dataset.rename_column ("conversation", "conversations")

# %%
roles = set ()
keys = set ()

for ex in dataset["conversations"]:
    for chat in ex:
        #print (chat["role"])
        roles.add (chat["from"]) # {'gpt', 'human'}

        for y in chat.keys():
            keys.add (y)

print (f"Roles do Dataset: {roles}")
print (f"Keys do Dataset: {keys}")

#dataset.to_parquet ("Dataset9.parquet")

# %%
def FULL_NORMALIZAR (dados):

    for x in dados["conversations"]:

        x["role"] = x.pop("from")
        x["content"] = x.pop("value")

    return dados

dataset = dataset.map (FULL_NORMALIZAR)

print (dataset[0])
print (dataset.features)

dataset.to_parquet ("Dataset10.parquet")

# %% [markdown]
# <hr>
# <h4> amalia-llm/wmt24pp_xx_to_pt_10k </h4>

# %%
dataset = load_dataset ("amalia-llm/wmt24pp_xx_to_pt_10k")["train"]

display (dataset.features)
display (dataset[0])
print (dataset)

# %%
roles = set ()
keys = set ()

for ex in dataset["conversations"]:
    for chat in ex:
        #print (chat["role"])
        roles.add (chat["from"]) # {'gpt', 'human'}

        for y in chat.keys():
            keys.add (y)

print (f"Roles do Dataset: {roles}")
print (f"Keys do Dataset: {keys}")

#dataset.to_parquet ("Dataset9.parquet")

# %%
def FULL_NORMALIZAR (dados):

    for x in dados["conversations"]:

        x["role"] = x.pop("from")
        x["content"] = x.pop("value")

    return dados

dataset = dataset.map (FULL_NORMALIZAR)

print (dataset[0])
print (dataset.features)

dataset.to_parquet ("Dataset12.parquet")

# %% [markdown]
# <h1> Dataset Final </h1>

# %%
dataset1 = load_dataset ("parquet", data_files = r"C:\Users\Admin\Desktop\MAIA\DatasetBuilder\Datasets\Dataset1.parquet", split = "train")
dataset2 = load_dataset ("parquet", data_files = r"C:\Users\Admin\Desktop\MAIA\DatasetBuilder\Datasets\Dataset2.parquet", split = "train")
dataset3 = load_dataset ("parquet", data_files = r"C:\Users\Admin\Desktop\MAIA\DatasetBuilder\Datasets\Dataset3.parquet", split = "train")
dataset4 = load_dataset ("parquet", data_files = r"C:\Users\Admin\Desktop\MAIA\DatasetBuilder\Datasets\Dataset4.parquet", split = "train")
dataset5 = load_dataset ("parquet", data_files = r"C:\Users\Admin\Desktop\MAIA\DatasetBuilder\Datasets\Dataset5.parquet", split = "train")
dataset6 = load_dataset ("parquet", data_files = r"C:\Users\Admin\Desktop\MAIA\DatasetBuilder\Datasets\Dataset6.parquet", split = "train")
dataset7 = load_dataset ("parquet", data_files = r"C:\Users\Admin\Desktop\MAIA\DatasetBuilder\Datasets\Dataset7.parquet", split = "train")
dataset8 = load_dataset ("parquet", data_files = r"C:\Users\Admin\Desktop\MAIA\DatasetBuilder\Datasets\Dataset8.parquet", split = "train")
dataset9 = load_dataset ("parquet", data_files = r"C:\Users\Admin\Desktop\MAIA\DatasetBuilder\Datasets\Dataset9.parquet", split = "train")
dataset10 = load_dataset ("parquet", data_files = r"C:\Users\Admin\Desktop\MAIA\DatasetBuilder\Datasets\Dataset10.parquet", split = "train")
dataset11 = load_dataset ("parquet", data_files = r"C:\Users\Admin\Desktop\MAIA\DatasetBuilder\Datasets\Dataset11.parquet", split = "train")


dataset = concatenate_datasets ([dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7, dataset8, dataset9, dataset10, dataset11])

display (dataset.features)
display (dataset[0])
print (dataset)

# %%
roles = set ()
keys = set ()

for ex in dataset["conversations"]:
    for chat in ex:
        #print (chat["role"])
        roles.add (chat["role"]) # {'gpt', 'human'}

        for y in chat.keys():
            keys.add (y)

print (f"Roles do Dataset: {roles}")
print (f"Keys do Dataset: {keys}")

dataset.to_parquet ("Dataset.parquet")


