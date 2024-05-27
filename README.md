# MaLoN : Multi-view attention LSTM for Network inference

This is the code for "Inference of gene regulatory networks for overcoming low performance in real world data"
This project is carried out in cooperation with [Funahashi Lab. at Keio University](https://fun.bio.keio.ac.jp/).


## Overview

Multi-view attention LSTM for Network inference (MaLoN) performs the task of gene regulatory network inference using time-series gene expression data as input.
An overview diagram of MaLoN is shown below.

![screenshot](https://github.com/funalab/MaLoN/raw/images/figure1.png)


## Requirements

- [Python 3.9](https://www.python.org/downloads/)
- [PyTorch 1.7+](https://chainer.org/)
- [SciPy 1.8+](https://scipy.org/)
- [Numpy](http://www.numpy.org)
- [pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Seaborn](https://seaborn.pydata.org/)

```sh
% pip install -r requirements.txt
```


## Quick Start

1. Download this repository by `git clone`.

```sh
% git clone https://github.com/funalab/MaLoN.git
```

2. Install requirements.

```sh
% cd MaLoN/
% python -m venv venv
% source ./venv/bin/activate
% pip install -r requirements.txt
```

3. Infer networks and evaluate the results for benchmark datasets.

To run MaLoN for reproduction in the article, follow the commands below. 

```sh
% cd src/tools/benchmarking
% ./inference_and_evaluation_for_invivo_datasets.sh mv
```

Or, run with ablated models.

```sh
% cd src/tools/benchmarking
% ./inference_and_evaluation_for_invivo_datasets_dnn.sh
% ./inference_and_evaluation_for_invivo_datasets_sv.sh
```

4. Infer networks for original datasets.

Note that input the format of expression data (row: gene, colomn: time-series, tab-deliminated, no rowname column)

4.1 Inference

```sh
% cd src/lib/models
% python main_mv.py [expression_file_path] --config [config_file_path] --name [dataset_name]
```

4.2 Evaluation (if you have the true network)

```sh
% cd src/lib/models
% python evaluation.py [predicted_edgelist_file_path] [true_edgelist_file_path] [dataset_name] [output_directory_path]
```

5. Visualize dynamic regulation maps.

Dynamic regulation maps were plotted when the number of genes is less than or equal to 10 in the 4.1.
If you want to plot them additionally or separately, follow the command below.

```sh
% cd src/lib/tools
% python plot_dynamic_regulation_map.py [dynamic regulation map file path] --type ['source' or 'target'] --out [output directory path] --name [dataset name]
```


# References

- [Van Anh Huynh-Thu and Guido Sanguinetti. Combining tree-based and dynamical systems for the inference of gene regulatory networks. Bioinformatics, Vol. 31, No. 10, pp. 1614–1622, 2015.](https://academic.oup.com/bioinformatics/article/31/10/1614/176842)
- [Van Anh Huynh-Thu and Pierre Geurts. dynGENIE3: dynamical GENIE3 for the inference of gene networks from time series expression data. Scientific reports, Vol. 8, No. 3384, 2018.](https://www.nature.com/articles/s41598-018-21715-0)
- [Ruiqing Zheng, Min Li, Xiang Chen, Fang-Xiang Wu, Yi Pan, and Jianxin Wang. BiXGBoost: a scalable, flexible boosting-based method for reconstructing gene regulatory networks. Bioinformatics, Vol. 35, No. 11, pp. 1893–1900, 2019.](https://academic.oup.com/bioinformatics/article/35/11/1893/5161079)
- [Baoshan Ma, Mingkun Fang, and Xiangtian Jiao. Inference of gene regulatory networks based on nonlinear ordinary differential equa- tions. Bioinformatics, Vol. 36, No. 19, pp. 4885–4893, 2020.](https://academic.oup.com/bioinformatics/article/36/19/4885/5709036)
- [Irene Cantone, Lucia Marucci, Francesco Iorio, Maria Aurelia Ricci, Vincenzo Belcastro, Mukesh Bansal, Stefania Santini, Mario Di Bernardo, Diego Di Bernardo, and Maria Pia Cosma. A yeast synthetic network for in vivo assessment of reverse-engineering and modeling approaches. Cell, Vol. 137, No. 1, pp. 172–181, 2009.](https://www.cell.com/fulltext/S0092-8674(09)00156-1)
- [Mohammad Yousef Memar, Mina Yekani, Giuseppe Celenza, Vahdat Poortahmasebi, Behrooz Naghili, Pierangelo Bellio, and Hossein Bannazadeh Baghi. The central role of the SOS DNA repair system in antibiotics resistance: A new target for a new infectious treatment strategy. Life Sciences, Vol. 262, 118562, 2020.](https://www.sciencedirect.com/science/article/pii/S0024320520313151)


# Acknowledgement

The development of this algorithm was funded by JST CREST Grant Number JPMJCR2011 including AIP challenge program to [Akira Funahashi](https://github.com/funasoul) and JSPS KAKENHI Grant Numbers 21J20961 to [Yusuke Hiki](https://github.com/hikiki-no-ki).
