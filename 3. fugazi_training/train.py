#https://gpuadvisor.com/compare
!pip install bitsandbytes
from typing import Literal
import torch #Torch não pode estar dentro da Classe porque é utilizada em dois métodos


class Training ():

  def __init__ (self, four_bit: bool, model: Literal ["Qwen/Qwen3-8B", "Qwen/Qwen3-4B", "Qwen/Qwen3-1.7B"] = "Qwen3-8B"):

    self.four_bit = four_bit
    self.model = model

    cuda = torch.cuda.is_available ()
    tpu = torch.accelerator.is_available ()

    if not tpu: #Não existe nenhum accelerator, ou seja = "cpu"
      raise RuntimeError ("Tem de existir uma GPU NVIDIA ou uma TPU!")
    if cuda and tpu: #Se existir cuda vai sempre existir accelerator, ou seja "cuda"
      self.device = "cuda"
    if tpu == True and cuda == False: #Se não existir uma GPU mas existir uma TPU, accelerator == True e cuda == False, ou seja "tpu"
      self.device = "tpu"

    if self.device == "cuda":
      if torch.cuda.is_bf16_supported ():
        self.w_type = torch.bfloat16
      else:
        self.w_type = torch.float16
    if self.device == "tpu":
      self.w_type = torch.bfloat16 #TPU's suportam sempre BF16

    if not isinstance (self.four_bit, bool):
      raise TypeError ("O argumento four_bit tem de ser um booleano!")

    if self.model not in ("Qwen/Qwen3-8B", "Qwen/Qwen3-4B", "Qwen/Qwen3-1.7B"):
      raise ValueError ("Modelo indisponível! Os únicos modelos suportados são [Qwen3-8B, Qwen3-4B, Qwen3-1.7B].")


  def Dados_Hardware (self):

    if self.device == "cuda":
      propriedades = torch.cuda.get_device_properties (device = self.device)
      mem_available = propriedades.total_memory / 1024**3

      print ("---------------------------------------------------- Dados GPU --------------------------------------------------------------")
      print (f"GPU: {propriedades.name}")
      print (f"Memória Disponível: {mem_available:.2f} GB")
      print (f"BF16 Supported: {torch.cuda.is_bf16_supported ()}")

      print (f"SM Count: {propriedades.multi_processor_count}")
      print (f"Memory Clock Rate: {propriedades.memory_clock_rate} kHz")
      print (f"Memory Bus Width: {propriedades.memory_bus_width} bits")
      print ("---" *42)

    if self.device == "tpu":
      propriedades = torch.tpu.get_device_properties ()
      mem_available = propriedades.total_memory / 1024**3

      print ("---------------------------------------------------- Dados TPU --------------------------------------------------------------")
      print (f"TPU: {propriedades.name}")
      print (f"Memória Disponível: {mem_available:.2f} GB")
      print ("---" *42)


  def Load_Model (self, engine: Literal ["unsloth+transformers", "transformers"] = "transformers"):

    try:
      from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as e:
      raise ImportError ("É necessário instalar as bibliotecas 'transformers' e 'bitsandbytes'.") from e

    if self.four_bit:
      try:
        BitsAndBytes_Config = BitsAndBytesConfig (load_in_4bit = True, bnb_4bit_quant_type = "nf4", bnb_4bit_compute_dtype = torch.float16, bnb_4bit_use_double_quant = True, bnb_4bit_quant_storage = torch.uint8)
        self.Tokenizer = AutoTokenizer.from_pretrained (self.model)
        self.Modelo = AutoModelForCausalLM.from_pretrained (self.model, device_map = self.device, dtype = self.w_type, quantization_config = BitsAndBytes_Config)
      except Exception as e:
        print (e)

    else:
      try:
        self.Tokenizer = AutoTokenizer.from_pretrained (self.model)
        self.Modelo = AutoModelForCausalLM.from_pretrained (self.model, device_map = self.device, dtype = self.w_type)
      except Exception as e:
        print (e)


  def Dados_Modelo (self):
    #print (self.Modelo.config)
    print ("---------------------------------------------------- Dados Modelo --------------------------------------------------------------")
    print (f"Nome do Modelo: {self.Modelo.config._name_or_path}")
    print (f"Formato dos pesos do Modelo: {self.Modelo.config.dtype}")
    print (f"Arquitetura do Modelo: {self.Modelo.config.architectures}")
    print (f"Hidden Size (Embedding Dimension): {self.Modelo.config.hidden_size}")
    print (f"Número Layers (Transformer Blocks): {self.Modelo.config.num_hidden_layers}")
    print (f"Attention Heads: {self.Modelo.config.num_attention_heads}")
    print (f"Vocab Size: {self.Modelo.config.vocab_size} tokens")
    print (f"Max Context: {self.Modelo.config.max_position_embeddings} tokens")
    print (f"Parâmetros: {self.Modelo.num_parameters () / 1e9:.2f} B")

    if hasattr (self.Modelo, "is_quantized"):
      print (f"Modelo Quantizado: {self.Modelo.is_quantized}")
      print (f"Método da Quantização : {self.Modelo.quantization_method}")

    print ("---------------------------------------------------- Dados Tokenizer --------------------------------------------------------------")
    print (f"Nome do Tokenizer: {self.Tokenizer.__class__.__name__}")


  def Load_Dataset (self, dataset_path: str, buffer_size: int):

    if buffer_size <= 0:
      raise ValueError ("'buffer_size' tem de ser um valor superior a 0.")
    
    try:
      from datasets import load_dataset
    except ImportError as e:
      raise ImportError ("É necessário instalar a biblioteca 'datasets'.") from e

    try:
      Dataset = load_dataset ("parquet", data_files = dataset_path, split = "train", streaming = True)
      self.dataset = Dataset.shuffle (buffer_size = buffer_size, seed = 42)
      #Sem Data Collator, explicar o porquê
    except Exception as e:
      print (e)


  def LoRA_Config (
      self, 
      proj_alvo: Literal ["q_proj", "k_proj", "v_proj", "o_proj"] = ["q_proj", "k_proj", "v_proj", "o_proj"], 
      rank: int = 16, 
      alpha: int = 32, 
      dropout: float = 0.07
      ):

    if rank <= 0:
      raise ValueError ("'rank' tem de ser um valor superior a 0.")
    if alpha <= 0:
      raise ValueError ("'alpha' tem de ser um valor suprior a 0.")
    if not 0.0 <= dropout < 1.0:
      raise ValueError ("'dropout' tem de ser um valor entre 0 e 1.")

    try:
      from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as e:
      raise ImportError ("É necessário instalar a biblioteca 'peft'.") from e

    if hasattr (self.Modelo, "is_quantized"):
      self.Modelo = prepare_model_for_kbit_training (self.Modelo)

    #https://huggingface.co/docs/peft/package_reference/lora
    LoRA = LoraConfig (
        target_modules = proj_alvo,
        r = rank,
        lora_alpha = alpha,
        lora_dropout = dropout,
        init_lora_weights = True,
        bias = "none",
        task_type = "CAUSAL_LM",
        use_rslora = True,
    )

    self.Modelo.config.use_cache = False
    self.Modelo = get_peft_model (self.Modelo, LoRA)
    print (self.Modelo.print_trainable_parameters ())


  def Argumentos_Treino (
      self, 
      steps_max: int, 
      batch_size: int, 
      effective_batch_size: int = 1, 

      optimizer: Literal ["adamw_torch", "adamw_bnb_8bit", "lora-fa", "lora+"] = "adamw_bnb_8bit",
      lr: float = 5e-5,       
      steps_warmup: int = 0,

      strat_eval: Literal ["no", "steps"] = "no",
      steps_eval: int | None = None, #int ou None, começando por None

      save: Literal ["no", "steps"] = "no",
      steps_save: int = 0,

      strat_log: Literal ["none", "steps"] = "no",
      steps_log: int = 1,

      workers_dataloader: int = 0,
      ):
  
    if steps_max <= 0 or batch_size <= 0 or steps_warmup <= 0:
      raise ValueError ("'steps_max', 'batch_size', 'steps_warmup' tem que ser um valor maior que 0.")
    if effective_batch_size <= 0:
      raise ValueError ("'effective_batch_size' tem que ser um valor maior que 0.")
    if optimizer not in ("adamw_torch", "adamw_bnb_8bit", "lora-fa", "lora+"):
      raise ValueError ("'optimizer' tem que ser um valor entre os seguintes: {adamw_torch, adamw_bnb_8bit, lora-fa ou lora+}")

    try:
      from transformers import TrainingArguments
    except ImportError as e:
      raise ImportError ("É necessário instalar a biblioteca 'transformers'.")

    BF16 = bool
    FP16 = bool
    if self.w_type is torch.bfloat16:
      BF16 = True
      FP16 = False
    else:
      FP16 = True
      BF16 = False

    self.optimizer = optimizer
    if self.optimizer == "lora-fa" or self.optimizer == "lora+":
      self.Args_Treino = TrainingArguments (
          #Treino
          do_train = True,
          max_steps = steps_max,
          per_device_train_batch_size = batch_size,
          gradient_accumulation_steps = effective_batch_size,

          #Scheduler
          lr_scheduler_type = "cosine",
          warmup_steps = steps_warmup, 
          weight_decay = 0.01,
          
          #Evaluation
          eval_strategy = strat_eval,
          eval_steps = steps_eval,

          #Checkpoint
          save_strategy = save,
          save_steps = steps_save,

          #Precision
          bf16 = BF16,
          fp16 = FP16,

          #Logging
          logging_strategy = strat_log,
          logging_steps = steps_log,

          #Dataloader
          dataloader_num_workers = workers_dataloader,
      )


    else:
      self.Args_Treino = TrainingArguments (
          #Treino
          do_train = True,
          max_steps = steps_max,
          per_device_train_batch_size = batch_size,
          gradient_accumulation_steps = effective_batch_size,

          #Optimizer
          optim = self.optimizer,
          learning_rate = lr,
          lr_scheduler_type = "cosine",
          warmup_steps = steps_warmup, 
          weight_decay = 0.01,
          
          #Evaluation
          eval_strategy = strat_eval,
          eval_steps = steps_eval,

          #Checkpoint
          save_strategy = save,
          save_steps = steps_save,

          #Precision
          bf16 = BF16,
          fp16 = FP16,          

          #Logging
          logging_strategy = strat_log,
          logging_steps = steps_log,

          #Dataloader
          dataloader_num_workers = workers_dataloader,
      )


  def Dados_Treino (self):
    print (self.Args_Treino)


  def Treino (self, engine: Literal ["unsloth, "transformers"] = "transformers"):






if __name__ == "__main__":

  DEMO = Training (four_bit = True, model = "Qwen/Qwen3-1.7B")
  #DEMO.Dados_Hardware ()
  #DEMO.Load_Model ()
  #DEMO.Dados_Modelo ()
  #DEMO.Argumentos_Treino (steps_max = 10, batch_size = 5, steps_warmup = 2, effective_batch_size = 3, optimizer = "adamw_bnb_8bit")
  #DEMO.Dados_Treino ()


