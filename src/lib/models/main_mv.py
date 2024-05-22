##########################################
# Project: Main script for network inference
# Author: Yusuke Hiki
##########################################

# Modules
import configparser
import time
import numpy as np
import torch
import tqdm
from multiprocessing import RLock
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn
import utils
from multi_view_attention_lstm_based_regressor import MultiViewAttentionLSTMBasedRegressor
# from multi_view_attention_dnn_based_regressor import MultiViewAttentionDNNBasedRegressor

def trainer(inputs):
    config_ini, input_file, i_target = inputs

    # Load input information from configulation file
    regulators_file = config_ini['Training']['REGULATORS_FILE']  # Default: all
    targets_file = config_ini['Training']['TARGETS_FILE']  # Default: all
    norm_method = config_ini['Training']['NORM_METHOD']  # Default: min_max
    num_layers = int(config_ini['Training']['NUM_LAYERS'])  # Default: 1
    hidden_size = int(config_ini['Training']['HIDDEN_SIZE'])  # Default: 10
    lr = float(config_ini['Training']['LEARNING_RATE'])  # Default: 0.1
    n_epoch = int(config_ini['Training']['N_EPOCH'])  # Default: 1000
    n_epoch_pretrain = 100  # Default: 100
    w_mse = float(config_ini['Training']['W_MSE'])  # Default: 1
    w_diff = float(config_ini['Training']['W_DIFF'])  # Default: 0
    w_attention = float(config_ini['Training']['W_ATTENTION'])  # Default: 0.01
    seed = int(config_ini['Other']['SEED'])  # Default: 1
    export_model = bool(config_ini['Other']['EXPORT_MODEL'] == 'True')  # Default: False
    output_dir = config_ini['Other']['OUTPUT_DIR']

    # Set seed for pytorch modules
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    # Get expression
    expression, gene_names = utils.load_datasets(input_file=input_file, input_format='tsv')
    expression = utils.normalization(expression=expression, norm_method=norm_method)

    i_source, _ = utils.get_gene_index(expression=expression,
                                       gene_name=gene_names,
                                       regulators_file=regulators_file,
                                       targets_file=targets_file)
    i_source = list(set(i_source) - set([i_target]))
    regulators_name = list(np.array(gene_names)[i_source])
    target_name = gene_names[i_target]

    # Get number of genes and number of timepoints
    n_regulators = len(i_source)
    n_timepoint = expression.shape[1]

    exp_source = torch.Tensor(expression[i_source, :n_timepoint-1])
    exp_target = torch.Tensor(expression[i_target][1:])

    model = MultiViewAttentionLSTMBasedRegressor(input_size=n_regulators,
                                                 hidden_size=hidden_size,
                                                 num_layers=num_layers)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001, weight_decay=0.01)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    mse = torch.nn.MSELoss()

    # Pretrain phase ---
    for epoch in range(n_epoch_pretrain):
        optimizer.zero_grad()

        # Regression
        exp_predicted_pretrain_i = model.pretrain(x=exp_source)

        # Get loss
        loss = mse(exp_predicted_pretrain_i, exp_target)

        # Backpropagation
        loss.backward()
        optimizer.step()

    print(gene_names[i_target] + "  Loss after pretraining: " + str(loss.data.item()))

    # Train (Inference) phase ---
    total_loss_list_i, mse_loss_list_i, diff_loss_list_i, attention_loss_list_i = [], [], [], []
    # info = f'{gene_names[i_target]:>2} '
    # for epoch in tqdm.tqdm(range(n_epoch), desc=info, position=i_target+1):
    for epoch in range(n_epoch):
        # print(str(epoch) + " epoch")
        optimizer.zero_grad()

        # Regression
        exp_predicted_i, dyn_reg_map_i, reg_score_i = model.forward(x=exp_source)

        # Get loss
        mse_loss = mse(exp_predicted_i, exp_target)
        diff_loss = utils.get_differential_loss(exp_predicted_i)
        attention_loss = utils.get_attention_loss(dyn_reg_map_i)
        total_loss = w_mse * mse_loss + w_diff * diff_loss + w_attention * attention_loss

        # Backpropagation
        total_loss.backward()

        optimizer.step()

        # Save loss
        total_loss_list_i.append(total_loss.data.item())
        mse_loss_list_i.append(mse_loss.data.item())
        diff_loss_list_i.append(diff_loss.data.item())
        attention_loss_list_i.append(attention_loss.data.item())

    loss_list_i = {'total_loss': total_loss_list_i,
                   'mse_loss': mse_loss_list_i,
                   'diff_loss': diff_loss_list_i,
                   'attention_loss': attention_loss_list_i}

    # Get edgelist for the target gene
    edgelist_i = utils.get_edgelist(source_name_list=regulators_name,
                                    target_name=target_name,
                                    scores=list(np.array(reg_score_i.data)))

    # Get regulation symbol (source->target)
    regulation_names = []
    for i in range(len(regulators_name)):
        regulation_names.append(regulators_name[i] + '->' + target_name)

    # Get dynamic regulation map (attention map) for the target gene
    dyn_reg_map_i = pd.DataFrame(np.array(dyn_reg_map_i.data), index=regulation_names)

    # Save model as pickle
    if export_model:
        output_model_dir = output_dir + "/" + DATASET_NAME
        output_model_filename = DATASET_NAME + "_" + gene_names[i_target] + "_targeted_epoch" + epoch + ".pickle"
        output_model_path = output_model_dir + "/" + output_model_filename
        torch.save(model, output_model_path)
        print("  Saved model:" + output_model_filename, end='')

    result_i = {'exp_predicted': np.array(exp_predicted_i.data),
                'dyn_reg_map': dyn_reg_map_i,
                'edgelist': edgelist_i,
                'loss_list': loss_list_i}

    return result_i


# Main
if __name__ == "__main__":
    # Load input information from arguments
    args = utils.arg_parser()
    INPUT_FILE = args.exp
    CFG_FILE = args.config
    DATASET_NAME = args.name
    MINOUT = args.minout

    config_ini = configparser.ConfigParser()
    config_ini.read(CFG_FILE, encoding='utf-8')
    REGULATORS_FILE = config_ini['Training']['REGULATORS_FILE']
    TARGETS_FILE = config_ini['Training']['TARGETS_FILE']
    OUTPUT_DIR = config_ini['Other']['OUTPUT_DIR']
    N_EPOCH = int(config_ini['Training']['N_EPOCH'])
    NORM_METHOD = config_ini['Training']['NORM_METHOD']

    # Set time measure
    start = time.time()

    # Get expression and number of genes
    expression, gene_names = utils.load_datasets(input_file=INPUT_FILE, input_format='tsv')
    expression = utils.normalization(expression=expression, norm_method=NORM_METHOD)

    regulators_index, targets_index = utils.get_gene_index(expression=expression,
                                                           gene_name=gene_names,
                                                           regulators_file=REGULATORS_FILE,
                                                           targets_file=TARGETS_FILE)
    regulators_name = list(np.array(gene_names)[regulators_index])
    targets_name = list(np.array(gene_names)[targets_index])

    n_target = len(targets_index)
    n_timepoint = expression.shape[1]

    # Inference in parallel
    input_data = [(config_ini, INPUT_FILE, i_target) for i_target in targets_index]

    # Inference with multiprocessing
    with torch.multiprocessing.Pool(processes=torch.multiprocessing.cpu_count(),
                                    initializer=tqdm.tqdm.set_lock,
                                    initargs=(RLock(), )) as pool:
        result = pool.map(trainer, input_data)

    # print("\n" * n_gene)

    # Extract result
    total_loss_list, mse_loss_list, diff_loss_list, attention_loss_list = [], [], [], []
    edgelist, dyn_reg_map = pd.DataFrame(), pd.DataFrame()
    for res in result:
        total_loss_list.append(res['loss_list']['total_loss'])
        mse_loss_list.append(res['loss_list']['mse_loss'])
        diff_loss_list.append(res['loss_list']['diff_loss'])
        attention_loss_list.append(res['loss_list']['attention_loss'])
        edgelist = pd.concat([edgelist, res['edgelist']])
        dyn_reg_map = pd.concat([dyn_reg_map, res['dyn_reg_map']])
        if 'exp_predicted' not in locals():
            exp_predicted = np.array(res['exp_predicted'].data)
        else:
            exp_predicted = np.vstack([exp_predicted, np.array(res['exp_predicted'].data)])

    # Reformat for save
    exp_predicted_pd = pd.DataFrame(exp_predicted, index=targets_name).T
    total_loss_list = pd.DataFrame(total_loss_list, index=targets_name).T
    mse_loss_list = pd.DataFrame(mse_loss_list, index=targets_name).T
    diff_loss_list = pd.DataFrame(diff_loss_list, index=targets_name).T
    attention_loss_list = pd.DataFrame(attention_loss_list, index=targets_name).T
    edgelist = edgelist.sort_values('score', ascending=False)
    edgelist.index = range(edgelist.shape[0])
    print(edgelist)

    # End measuring time
    elapsed_time = time.time() - start
    print("Elapsed time: " + str(elapsed_time) + " s")

    # Add directory if not exist
    output_ds_dir = OUTPUT_DIR + "/" + DATASET_NAME + "/"
    if not os.path.exists(output_ds_dir):
        os.makedirs(output_ds_dir)

    # Output edgelist, loss list, and dynamic regulation map
    edgelist.to_csv(output_ds_dir + DATASET_NAME + "_edgelist.tsv", sep='\t', index=False)
    dyn_reg_map.to_csv(output_ds_dir + DATASET_NAME + "_dynamic_regulation_map.tsv", sep='\t', header=False)

    if MINOUT is False:
        exp_predicted_pd.to_csv(output_ds_dir + DATASET_NAME + "_regressed_expression.tsv", sep='\t', index=False)
        total_loss_list.to_csv(output_ds_dir + DATASET_NAME + "_loss.tsv", sep='\t', index=False)
        mse_loss_list.to_csv(output_ds_dir + DATASET_NAME + "_mse_loss.tsv", sep='\t', index=False)
        diff_loss_list.to_csv(output_ds_dir + DATASET_NAME + "_diff_loss.tsv", sep='\t', index=False)
        attention_loss_list.to_csv(output_ds_dir + DATASET_NAME + "_attention_loss.tsv", sep='\t', index=False)

        # Plot Train curve (only when less than 10 genes)
        if len(targets_name) <= 10:
            cmap = plt.get_cmap("tab10")
            plt.figure()
            for i in range(n_target):
                plt.plot(range(N_EPOCH), total_loss_list.iloc[:, i], color=cmap(i), label=targets_name[i], linestyle="solid")
            plt.xlabel("Epoch", fontsize=16)
            plt.ylabel("Loss", fontsize=16)
            plt.title("Train loss curve")
            plt.grid(True)
            plt.legend()
            pp1 = PdfPages(output_ds_dir + DATASET_NAME + "_train_loss.pdf")
            pp1.savefig()
            pp1.close()
            plt.close()

        # Plot regressed expression (only when less than 500 genes)
        if len(targets_name) <= 500:
            for i in range(len(targets_index)):
                plt.figure(figsize=(8, 4))
                plt.scatter(range(0, n_timepoint), expression[targets_index[i]], color="red", label="Measured")
                plt.plot(range(1, n_timepoint), exp_predicted[i], color="blue", label="Predicted",
                         marker=".", linestyle="dotted")
                plt.xlabel("Time", fontsize=16)
                plt.ylabel("Expression", fontsize=16)
                plt.title("Expression curve (" + targets_name[i] + ")")
                plt.grid(True)
                plt.legend(loc="upper right")
                pp1 = PdfPages(output_ds_dir + DATASET_NAME + "_" + targets_name[i] + "_regression_result.pdf")
                pp1.savefig()
                pp1.close()
                plt.close()

        # Plot dynamic regulation map (only when less than 10 genes)
        if len(targets_name) <= 10:
            plt.figure(figsize=(12, 12))  # This size should be automatically optimized
            seaborn.heatmap(dyn_reg_map, cmap="Reds", linewidths=0.01, linecolor='grey', yticklabels=True)
            plt.figure(figsize=(12, 4))
            seaborn.heatmap(dyn_reg_map, cmap="Reds", linewidths=0.01, linecolor='grey')
            pp2 = PdfPages(output_ds_dir + DATASET_NAME + "_dynamic_regulation_map.pdf")
            pp2.savefig()
            pp2.close()
            plt.close()
