#https://gpuadvisor.com/compare

!pip install bitsandbytes
!pip install unsloth

from typing import Literal
import torch 
from transformers import TrainerCallback

import time
import psutil
import logging
import numpy as np
import yaml
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates
)

class Treino_Log (TrainerCallback):

    def __init__ (self, dataset_card: str):

      cuda = torch.cuda.is_available ()

      with open (dataset_card, "r") as file:
        self.dataset_card = yaml.safe_load (file)

      logging.basicConfig (filename = "Log_Treino.log", level = logging.INFO, format = "%(message)s", force = True) #Cria um log limpo
      self.log = logging.getLogger (__name__)

      self.Metricas = {
          "CPU_MEM": [],
          "CPU_UTIL": [],
          "TEMPO_STEP": [],
          "TOKEN_PER_S": [],
          "GPU_MEM": [],
          "TORCH_MEM_GPU_ALLOC": [],
          "TORCH_MEM_GPU_RESERV": [],
          "UTIL_RATE_MEM_GPU": [], #Memória da GPU
          "UTIL_RATE_GPU": [] #Processamento da GPU
      }

      nvmlInit ()
      if cuda:
        self.GPU = nvmlDeviceGetHandleByIndex (0)


    def on_train_begin (self, args, state, control, **kwargs):

      torch.cuda.empty_cache ()
      torch.cuda.reset_peak_memory_stats ()

      self.log.info ("Training Arguments: %s", args.to_dict ())

    def on_step_begin (self, args, state, control, **kwargs):

      self.log.info ("Forward Pass %s", state.global_step)

      self.STEP_START_TIME = time.perf_counter ()

    def on_substep_end (self, args, state, control, **kwargs):

      MEM = nvmlDeviceGetMemoryInfo (self.GPU)

      self.Metricas ["GPU_MEM"].append (MEM.used / 1024**3) #GB
      self.Metricas ["TORCH_MEM_GPU_ALLOC"].append (torch.cuda.memory_allocated () / 1024**3) #GB
      self.Metricas ["TORCH_MEM_GPU_RESERV"].append (torch.cuda.memory_reserved () / 1024**3) #GB
      self.Metricas ["UTIL_RATE_MEM_GPU"].append (MEM.used / MEM.total * 100) #%
      self.Metricas ["UTIL_RATE_GPU"].append (nvmlDeviceGetUtilizationRates (self.GPU).gpu) #%

      self.Metricas ["CPU_MEM"].append (psutil.virtual_memory ().used / 1024**3) #GB
      self.Metricas ["CPU_UTIL"].append (psutil.cpu_percent (interval = None)) #%

    def on_step_end (self, args, state, control, **kwargs):

      GPU_MEM_USADA = nvmlDeviceGetMemoryInfo (self.GPU).used / 1024**3 #GB
      GPU_MEM_TORCH_ALLOC = torch.cuda.memory_allocated () / 1024**3 #GB
      GPU_MEM_TORCH_RESERV = torch.cuda.memory_reserved () / 1024**3 #GB

      self.log.info ("Memória GPU nvml: %.2f GB", GPU_MEM_USADA) #GB
      self.log.info ("Memória Torch GPU Alocada: %.2f GB", GPU_MEM_TORCH_ALLOC) #GB
      self.log.info ("Memória Torch GPU Reservada: %.2f GB", GPU_MEM_TORCH_RESERV) #GB

      CPU_MEM = psutil.virtual_memory ()
      self.log.info ("Memória CPU Usada: %.2f GB", CPU_MEM.used / 1024**3) #GB

      TEMPO_STEP = time.perf_counter () - self.STEP_START_TIME
      self.Metricas ["TEMPO_STEP"].append (TEMPO_STEP) #Segundos
      self.log.info ("Tempo do Step: %s s", TEMPO_STEP) #Segundos

      TOKEN_PER_S = self.dataset_card ["Sequence Length"] * args.per_device_train_batch_size * args.gradient_accumulation_steps / TEMPO_STEP
      self.Metricas ["TOKEN_PER_S"].append (TOKEN_PER_S)
      self.log.info ("Tokens Per Second: %s tok/s", TOKEN_PER_S)


    def on_train_end (self, args, state, control, **kwargs):

      self.log.info ("Estatísticas do Treino Realizado")
      for dados in state.log_history:
        self.log.info ("%s", dados)

      self.log.info ("Estatísticas do Treino Realizado")

      self.log.info ("Logs Memória GPU NVML: %s", self.Metricas ["GPU_MEM"])
      self.log.info ("Max Logs Memória GPU NVML: %s GB", np.max (self.Metricas ["GPU_MEM"]))
      self.log.info ("Mean Logs Memória GPU NVML: %s GB", np.mean (self.Metricas ["GPU_MEM"]))

      self.log.info ("Logs Memória Torch GPU Alocada: %s", self.Metricas ["TORCH_MEM_GPU_ALLOC"])
      self.log.info ("Max Logs Memória Torch GPU Alocada: %s GB", np.max (self.Metricas ["TORCH_MEM_GPU_ALLOC"]))
      self.log.info ("Mean Logs Memória Torch GPU Alocada: %s GB", np.mean (self.Metricas ["TORCH_MEM_GPU_ALLOC"]))

      self.log.info ("Logs Memória Torch GPU Reservada: %s", self.Metricas ["TORCH_MEM_GPU_RESERV"])
      self.log.info ("Max Logs Memória Torch GPU Reservada: %s GB", np.max (self.Metricas ["TORCH_MEM_GPU_RESERV"]))
      self.log.info ("Mean Logs Memória Torch GPU Reservada: %s GB", np.mean (self.Metricas ["TORCH_MEM_GPU_RESERV"]))

      self.log.info ("Logs Percentagem de Utilização da Memória GPU: %s", self.Metricas ["UTIL_RATE_MEM_GPU"])
      self.log.info ("Max Logs Percentagem de Utilização da Memória GPU: %s %", np.max (self.Metricas ["UTIL_RATE_MEM_GPU"]))
      self.log.info ("Mean Logs Percentagem de Utilização da Memória GPU: %s %", np.mean (self.Metricas ["UTIL_RATE_MEM_GPU"]))

      self.log.info ("Logs Percentagem de Utilização da GPU: %s", self.Metricas ["UTIL_RATE_GPU"])
      self.log.info ("Max Logs Percentagem de Utilização da GPU: %s %", np.max (self.Metricas ["UTIL_RATE_GPU"]))
      self.log.info ("Mean Logs Percentagem de Utilização da GPU: %s %", np.mean (self.Metricas ["UTIL_RATE_GPU"]))

      self.log.info ("Logs Memória CPU: %s", self.Metricas ["CPU_MEM"])
      self.log.info ("Max Logs Memória CPU: %s GB", np.max (self.Metricas ["CPU_MEM"]))
      self.log.info ("Mean Logs Memória CPU: %s GB", np.mean (self.Metricas ["CPU_MEM"]))

      self.log.info ("Logs Percentagem de Utilização da CPU: %s", self.Metricas ["CPU_UTIL"])
      self.log.info ("Max Logs Percentagem de Utilização da CPU: %s %", np.max (self.Metricas ["CPU_UTIL"]))
      self.log.info ("Mean Logs Percentagem de Utilização da CPU: %s %", np.mean (self.Metricas ["CPU_UTIL"]))

      self.log.info ("Logs Tempo Por Step: %s", self.Metricas ["TEMPO_STEP"])
      self.log.info ("Logs Tempo Por Step: %s s", np.max (self.Metricas ["TEMPO_STEP"]))
      self.log.info ("Logs Tempo Por Step: %s s", np.mean (self.Metricas ["TEMPO_STEP"]))

      self.log.info ("Logs Tokens Per Second: %s", self.Metricas ["TOKEN_PER_S"])
      self.log.info ("Max Logs Tokens Per Second: %s tokens/s", np.max (self.Metricas ["TOKEN_PER_S"]))
      self.log.info ("Mean Logs Tokens Per Second: %s tokens/s", np.mean (self.Metricas ["TOKEN_PER_S"]))




class Training ():

  def __init__ (
      self,
      four_bit: bool,
      model: Literal ["Qwen/Qwen3-8B", "Qwen/Qwen3-4B", "Qwen/Qwen3-1.7B"] = "Qwen/Qwen3-4B"
      ):

    self.device = "cuda" if torch.cuda.is_available () else "cpu"

    if self.device != "cuda":
      raise RunTimeError ("NO CUDA, NO PARTY!")

    self.four_bit = four_bit
    self.model = model

    capacidade = torch.cuda.get_device_capability()
    if capacidade >= (8, 0) and torch.cuda.is_bf16_supported():
      self.w_type = torch.bfloat16
    else:
      self.w_type = torch.float16

    ##Arquitetura Volta - 7.0 (FP16) | Turing - 7.5 (FP16) | Ampere - 8.0 (BF16)

    if not isinstance (self.four_bit, bool):
      raise TypeError ("O argumento four_bit tem de ser um booleano!")

    if self.model not in ("Qwen/Qwen3-8B", "Qwen/Qwen3-4B", "Qwen/Qwen3-1.7B"):
      raise ValueError ("Modelo indisponível! Os únicos modelos suportados são [Qwen3-8B, Qwen3-4B, Qwen3-1.7B].")


  def Dados_Hardware (self):

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


  def Load_Model_Transformers (self):

    self.engine_model = "Transformers"
    try:
      from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as e:
      raise ImportError ("É necessário instalar as bibliotecas 'transformers' e 'bitsandbytes'.") from e

    if self.four_bit:
      try:
        BitsAndBytes_Config = BitsAndBytesConfig (load_in_4bit = True, bnb_4bit_quant_type = "nf4", bnb_4bit_compute_dtype = self.w_type, bnb_4bit_use_double_quant = True, bnb_4bit_quant_storage = torch.uint8)
        self.Tokenizer = AutoTokenizer.from_pretrained (self.model)
        self.Modelo = AutoModelForCausalLM.from_pretrained (self.model, device_map = self.device, dtype = self.w_type, quantization_config = BitsAndBytes_Config)
      except Exception as e:
        raise RunTimeError (f"Falha ao carregar o modelo {self.Modelo}!")

    else:
      try:
        self.Tokenizer = AutoTokenizer.from_pretrained (self.model)
        self.Modelo = AutoModelForCausalLM.from_pretrained (self.model, device_map = self.device, dtype = self.w_type)
      except Exception as e:
        raise RunTimeError (f"Falha ao carregar o modelo {self.Modelo}!")


  def Load_Model_Unsloth (self):

    self.engine_model = "Unsloth"
    try:
      from unsloth import FastLanguageModel
    except ImportError as e:
      raise ImportError ("É necessário instalar a biblioteca 'unsloth'.")

    if self.four_bit:
      try:
        self.Modelo, self.Tokenizer = FastLanguageModel.from_pretrained (model_name = self.model, dtype = self.w_type, load_in_4bit = True)
      except Exception as e:
        print (e)

    else:
      try:
        self.Modelo, self.Tokenizer = FastLanguageModel.from_pretrained (model_name = self.model, dtype = self.w_type, load_in_4bit = False, load_in_8bit = False)
      except Exception as e:
        print (e)


  def Dados_Modelo (self):
    #print (self.Modelo.config)
    print ("---------------------------------------------------- Dados Modelo --------------------------------------------------------------")
    print (f"Nome do Modelo: {self.Modelo.config._name_or_path}")
    print (f"Formato dos Cálculos do Modelo: {self.Modelo.config.dtype}")
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

    self.r = rank
    self.a = alpha

    if self.engine_model == "Transformers":
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
      self.Modelo.print_trainable_parameters ()

    if self.engine_model == "Unsloth":
      try:
        from unsloth import FastLanguageModel
      except ImportError as e:
        raise ImportError ("É necessário instalar a biblioteca 'unsloth'.")

      self.Modelo = FastLanguageModel.get_peft_model (
          self.Modelo,
          target_modules = proj_alvo,
          r = rank,
          lora_alpha = alpha,
          lora_dropout = dropout,
          bias = "none",
          use_rslora = True,
      )

      self.Modelo.config.use_cache = False
      self.Modelo.print_trainable_parameters ()


  def Load_Dataset (self, dataset_path: str, buffer_size: int, dataset_card: str):

    if buffer_size <= 0:
      raise ValueError ("'buffer_size' tem de ser um valor superior a 0.")

    try:
      from datasets import load_dataset
      import yaml
    except ImportError as e:
      raise ImportError ("É necessário instalar a biblioteca 'datasets'.") from e

    try:
      Dataset = load_dataset ("parquet", data_files = dataset_path, split = "train", streaming = True)
      self.dataset = Dataset.shuffle (buffer_size = buffer_size, seed = 42)

      with open (dataset_card, "r") as file:
        self.dataset_card = yaml.safe_load (file)

      #Sem Data Collator, explicar o porquê
    except Exception as e:
      print (e)


  def Argumentos_Treino (
      self, 
      steps_max: int = 10, 
      batch_size: int = 1, 
      effective_batch_size: int = 1, 
      steps_warmup: int = 0, 
      
      strat_eval: Literal ["no", "steps"] = "no", 
      steps_eval: int | None = None, #int ou None, começando por None
      save: Literal ["no", "steps"] = "no",
      
      steps_save: int = 0,
      strat_log: Literal ["no", "steps"] = "no",
      steps_log: int = 1,

      ):

    try:
      from transformers import TrainingArguments
    except ImportError as e:
      raise ImportError ("É necessário instalar a biblioteca 'transformers'.")

    BF16 = self.w_type == torch.bfloat16
    FP16 = self.w_type == torch.float16
  
    self.Args_Treino = TrainingArguments (
          #Treino
        do_train = True,
        max_steps = steps_max,
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = effective_batch_size,

          #Optimizer
        optim = "adamw_bnb_8bit",
        learning_rate = 2e-4,
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

      )


  def Dados_Treino (self):

    EXEMPLOS = self.dataset_card ["Exemplos"]

    print ("---------------------------------------------------- Dados Treino --------------------------------------------------------------")
    print (f"Número de Exemplos Processados num Step: {self.Args_Treino.per_device_train_batch_size * self.Args_Treino.gradient_accumulation_steps}")
    print (f"Steps Necessários para completar 1 Epoch: {EXEMPLOS / (self.Args_Treino.per_device_train_batch_size * self.Args_Treino.gradient_accumulation_steps)}")

    print (self.Args_Treino)


  def Treino (self, Treino_Log):

    try:
      from transformers import Trainer
    except Exception as e:
      print (e)

    #Begin Training
    trainer = Trainer (
        model = self.Modelo,
        args = self.Args_Treino,
        train_dataset = self.dataset,
        callbacks = [Treino_Log]
    )

    trainer.train ()



if __name__ == "__main__":

  DEMO = Training (four_bit = True, model = "Qwen/Qwen3-1.7B")
  DEMO.Dados_Hardware ()

  #DEMO.Load_Model_Transformers ()
  DEMO.Load_Model_Unsloth ()
  DEMO.Dados_Modelo ()

  DEMO.LoRA_Config ()

  DEMO.Load_Dataset (dataset_path = "/content/Dataset_Mean.parquet", buffer_size = 1000, dataset_card = "/content/Dataset_Mean_Card.yaml")

  DEMO.Argumentos_Treino (steps_max = 10, batch_size = 5, steps_warmup = 2, effective_batch_size = 3)
  DEMO.Dados_Treino ()

  DEMO.Treino (Treino_Log (dataset_card = "/content/Dataset_Mean_Card.yaml"))