<h1> Fugazi Datasets </h1>

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