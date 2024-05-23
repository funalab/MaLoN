# MaLoN (Multi-view attention LSTM for Network inference)

## Requirement

- python >=3.9.18
- zsh 5.8.1 (x86_64-apple-darwin21.0)
- python modules in the ./requirements/requirements.txt

```sh
% cd ./requirements/
% pip install -r requirements.txt
```


## How to test

1. Access to benchmark datasets

Refer to the README in the following directory to get it.

- Ecoli_SOSPathway: ./src/lib/datasets/ecoli_sospathway_datasets/
- IRMA: ./src/lib/datasets/irma_datasets/

2. Inference and evaluation

Infer networks and evaluate the results for benchmark datasets

```sh
% cd src/tools/benchmarking
% ./inference_and_evaluation_for_invivo_datasets.sh mv
```

With ablated models

```sh
% cd src/tools/benchmarking
% inference_and_evaluation_for_invivo_datasets_dnn.sh
```


## How to use for original datasets

Note that input the format of expression data (row: gene, colomn: time-series, tab-deliminated, no rowname column)

Inference
```sh
% cd src/lib/models
% python main_mv.py [expression_file_path] --config [config_file_path] --name [dataset_name]
```

Evaluation (if you have the true network)
```sh
% cd src/lib/models
% python evaluation.py [predicted_edgelist_file_path] [true_edgelist_file_path] [dataset_name] [output_directory_path]
```


## Abstraction of algorithm
![screenshot](https://github.com/funalab/MaLoN/raw/images/figure1.png)
