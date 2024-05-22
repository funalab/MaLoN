##########################################
# Project: functions as tools for network inference
# Author: Yusuke Hiki
##########################################

# Modules
import numpy as np
import torch
import pickle
import sys
import pandas as pd
import argparse

# Function to parse argument
def arg_parser():
    parser = argparse.ArgumentParser(description='Gene regulatory network inference with multi-view attention LSTM')
    parser.add_argument('exp', help='Input file of time-series gene expression data')
    parser.add_argument('--config',
                        help='Config file to decide hyper parameters',
                        default='../../../confs/multi_view.cfg')
    parser.add_argument('--name',
                        help='Dataset name for prefix',
                        default='multi_view_attention_lstm')
    parser.add_argument('--minout',
                        help='Output dynamic regulation map and edgelist only or not',
                        default=False)

    args = parser.parse_args()
    return args


# Function to load datasets from pickle files
def load_datasets(input_file, input_format='tsv'):

    # Load files of tsv format
    if (input_format == 'tsv') or (input_format == 'txt'):
        exp_data = pd.read_csv(input_file, sep='\t')
        gene_name_list = list(exp_data.columns)
        exp_data = np.array(exp_data.T)

    # Load files of csv format
    elif input_format == 'csv':
        exp_data = pd.read_csv(input_file, sep=',')
        gene_name_list = list(exp_data.columns)
        exp_data = np.array(exp_data.T)

    # Load files of pickle format
    elif input_format == 'pickle':
        f_data = open(input_file, 'rb')
        data = pickle.load(f_data)
        f_data.close
        exp_data, adj_mat_data = data[0], data[1]
        gene_name_list = list(exp_data.columns)
        exp_data, adj_mat_data = np.array(exp_data).T, np.array(adj_mat_data)

    # Check format validity
    else:
        print("ERROR: Input file format is invalid. designate 'tsv', 'txt', 'csv', or 'pickle'")
        sys.exit()

    return exp_data, gene_name_list


# Function to normalize expression data
def normalization(expression, norm_method='zscore'):
    if norm_method == "min_max":
        expression = min_max_normalization_exp(exp_data=expression)
    elif norm_method == "zscore":
        expression = zscore_normalization_exp(exp_data=expression)
    elif norm_method != "raw":
        print("ERROR: NORM_METHOD is invalid. set 'raw', 'min_max', or 'zscore'")
        sys.exit()

    return expression


# Function to normalize expression data to z-score
def zscore_normalization_exp(exp_data):
    exp_mean = np.mean(exp_data, axis=1)
    exp_std = np.std(exp_data, axis=1)
    exp_zscore = np.zeros(exp_data.shape)
    for i in range(exp_data.shape[0]):
        exp_zscore_i = (exp_data[i] - exp_mean[i])/exp_std[i]
        exp_zscore[i] = exp_zscore_i

    return exp_zscore


# Function to normalize expression data as min-max is 0~1
def min_max_normalization_exp(exp_data):
    exp_min = np.min(exp_data, axis=1)
    exp_max = np.max(exp_data, axis=1)
    range_v = exp_max - exp_min
    exp_min_max = np.zeros(exp_data.shape)
    for i in range(exp_data.shape[0]):
        if range_v[i] > 0:
            exp_min_max_i = (exp_data[i] - exp_min[i])/range_v[i]
            exp_min_max[i] = exp_min_max_i
        else:
            exp_min_max[i] = 0

    return exp_min_max


# Function to normalize vector as min-max is 0~1
def min_max_normalization_vec(vec):
    min_v = torch.min(vec)
    range_v = torch.max(vec) - min_v
    if range_v > 0:
        normalized = (vec - min_v) / range_v
    else:
        normalized = torch.zeros(vec.shape)

    return normalized


# Function to get difference in time-series for differential L1 loss
def get_differential_loss(vec_tensor):
    return torch.mean(torch.abs(vec_tensor[1:] - vec_tensor[0:(-1)]))


# Function to get loss as attention loss
def get_attention_loss(mat_tensor):
    return torch.mean(1/torch.sum(torch.square(mat_tensor), dim=0) - 1)


# Function to get loss as attention loss
def get_attention_loss2(mat_tensor):
    return torch.mean(1/torch.sum(torch.square(mat_tensor), dim=0) - 1)


# Function to generate edgelist
def get_edgelist(source_name_list, target_name, scores):
    el = []
    for i in range(len(source_name_list)):
        el_i = [source_name_list[i], target_name, scores[i]]
        el.append(el_i)

    el = pd.DataFrame(el, columns=['source', 'target', 'score'])

    return el


# Function to get regulators and target genes index
def get_gene_index(expression, gene_name, regulators_file='all', targets_file='all'):
    if regulators_file == 'all':
        regulators_index = list(set(range(expression.shape[0])))
    else:
        regulators = pd.read_csv(regulators_file, header=None)
        regulators = regulators[0].values.tolist()
        regulators_index = [gene_name.index(i) for i in regulators]

    if targets_file == 'all':
        targets_index = list(set(range(expression.shape[0])))
    else:
        targets = pd.read_csv(targets_file, header=None)
        targets = targets[0].values.tolist()
        targets_index = [gene_name.index(i) for i in targets]

    return regulators_index, targets_index

'''
def random_embedding(x):
    xy_size = [10, 10]
    x = np.zeros(xy_size)
    y = np.ones(xy_size)

    x.
'''
