<h1> Versão TPU </h1>

Houve uma tentativa da minha parte de construir um <b> wrapper </b> à volta das librarys <b>Transformers</b>, <b>Unsloth</b> e <b> Torch</b> para criar uma classe que poderia treinar modelos tanto em GPU como em TPU. <br>
Apesar dos meus esforços, o suporte para TPU ainda é muito limitado, contudo deixo aqui a versão que construí porque pode ser que no futuro seja relevante.

<h1> Versão GPU </h1>

Esta versão é a versão final, e <b>a que vai ser usada no treino</b>. Tem suporte a GPU's NVIDIA usando o ecosistema <b>Transformers</b>, <b>Unsloth</b> e <b> Torch </b>.

<hr>

<a href = "https://github.com/huggingface/transformers">Transformers - https://github.com/huggingface/transformers</a> <br>
<a href = "https://github.com/unslothai/unsloth">Unsloth - https://github.com/unslothai/unsloth</a><br>
<a href = "https://github.com/pytorch/pytorch">PyTorch - https://github.com/pytorch/pytorch</a><br>