from transformers import AutoTokenizer
from datasets import load_dataset

"""
Na construção deste .py deparei-me com uma questão que é muito importante ter em conta.
Ao realizar a Tokenização para analisar o método Estimate Training Cost Based on Sequence Length Distribution estava a apenas a pensar em realizar Tokenização aos exemplos concretos ou 
seja "content", porém isto é errado para esta metodologia.
Se a ideia é ter uma previsão do tempo, memória etc do treino, é preciso realizar a Tokenização como se fosse para o dataset final de treino.
Temos então que aplicar chat template e depois realizar tokenização de tudo porque é isso que acontece no treino original. Se fizessemos como estava a pensar, era errado porque não estaríamos
a cobrir o cenário real de treino.
"""

#### Tokenizer
MODEL = "Qwen/Qwen3-8B"
TOKENIZER = AutoTokenizer.from_pretrained (MODEL)

#### Dataset
dataset = load_dataset ("parquet", data_files = r"C:\Users\Admin\Desktop\MAIA\DatasetBuilder\Datasets\Dataset.parquet", split = "train")

#teste = load_dataset ("parquet", data_files = r"C:\Users\Admin\Desktop\MAIA\DatasetBuilder\Datasets\Dataset.parquet", split = "train[:1%]")

def TOKENIZAÇÃO (dados):
  
    """
    Sem Truncation e sem Padding! Queremos a Tokenização pura e crua.
        
    # Chat Template # https://huggingface.co/docs/transformers/chat_templating # Importante perceber os params que podem ser passados.
    # add_generation_prompt = False porque não queremos que a Tokenização adicione <assistant> no fim, isso é mais para inferência.
    # enable_thinking = False - Erro detetado no dia 01/09 onde o dataset tokenizado estava a produzir <think> </think> pois o modelo Qwen tem a opção de Thinking, porém
    para este treino não temos um dataset preparado para thinking.

    01/09 - Cheguei à conclusão que não importa ter <think> </think> ou não de acordo com este link:
    https://huggingface.co/blog/qwen-3-chat-template-deep-dive - Para perceber enable_thinking
    """

    EXEMPLO = TOKENIZER.apply_chat_template (dados["conversations"], tokenize = False, add_generation_prompt = False, enable_thinking = False) 
 
    TOKENS = TOKENIZER (EXEMPLO, add_special_tokens = False) # Tokenização

    return TOKENS

dataset_tokenizado = dataset.map (TOKENIZAÇÃO, remove_columns = "conversations")

dataset_tokenizado.to_parquet ("DatasetTokenizado.parquet")











"""
ISTO SERIA O CORRETO CASO A IDEIA FOSSE TOKENIZAR APENAS O CONTENT! 

for exemplo in dataset["conversations"][:3]:
    for conversa in exemplo:
        #print (conversa["content"])
        print (TOKENIZER (conversa["content"], add_special_tokens = True))
"""