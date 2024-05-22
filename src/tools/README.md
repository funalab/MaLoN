- `plot_dynamic_regulation_map.py`

Plot dynamic regulation map for each gene as target or source from inferred dynamic regulation map matrix data

```
% python plot_dynamic_regulation_map.py [dynamic regulation map file path] --type ['source' or 'target'] --out [output directory path] --name [dataset name]
```


- `benchmarking/inference_and_evaluation_forinvivo_datasets.sh`

Infer networks and evaluate the performance of MaLoN using benchmark datasets

```
% cd benchmarking
% ./inference_and_evaluation_for_invivo_datasets.sh
```


- `benchmarking/inference_and_evaluation_forinvivo_datasets_dnn.sh`

Infer networks and evaluate the performances of Mv-DNN10 and Mv-DNN36 using benchmark datasets

```
% cd benchmarking
% ./inference_and_evaluation_for_invivo_datasets_dnn.sh
```


- `benchmarking/inference_and_evaluation_forinvivo_datasets_sv.sh`

Infer networks and evaluate the performance of Sv-LSTM using benchmark datasets

```
% cd benchmarking
% ./inference_and_evaluation_for_invivo_datasets_sv.sh
```
