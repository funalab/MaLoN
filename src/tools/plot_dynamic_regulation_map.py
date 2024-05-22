##########################################
# Project: Plot dynamic regulation map
# Author: Yusuke Hiki
##########################################

# Modules
import sys
import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn


# Function to parse argument
def arg_parser_plot_map():
    parser = argparse.ArgumentParser(description='Plot inferred dynamic regulation map')
    parser.add_argument('map', help='Input file of dynamic regulation map data')
    parser.add_argument('--type', help='Focused on specific gene as source or target', default='target')
    parser.add_argument('--out', help='Output directory path', default='./dynamic_regulation_map/')
    parser.add_argument('--name', help='Dataset name', default='mydata')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Load input information from arguments
    args = arg_parser_plot_map()
    INPUT_FILE = args.map
    REG_TYPE = args.type
    OUTPUT_DIR = args.out
    DATASET_NAME = args.name

    # Load dynamic regulation map data
    dyn_reg_map = pd.read_csv(INPUT_FILE, sep='\t', header=None, index_col=0)
    dyn_reg_map = dyn_reg_map.rename_axis('')

    # Get target gene name set in regulations
    reg_name_list = dyn_reg_map.index.to_list()

    # Plot regulation map TO each gene
    if REG_TYPE == 'target':
        reg_target = [x.split('->')[1] for x in reg_name_list]
        target_name = list(sorted(set(reg_target)))

        # Add directory if not exist
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Plot dynamic regulation map (only when less than 10 genes)
        for i in range(len(target_name)):
            print(target_name[i])
            target_index = [j for j, x in enumerate(reg_target) if x == target_name[i]]
            dyn_reg_map_target = dyn_reg_map.iloc[target_index, :]

            plt.figure(figsize=(12, 12))  # This size should be automatically optimized
            seaborn.heatmap(dyn_reg_map_target, cmap="Reds", linewidths=0.01, linecolor='grey', yticklabels=True)
            pp = PdfPages(OUTPUT_DIR + "/" + DATASET_NAME + "_to_" + target_name[i] + "_dynamic_regulation_map.pdf")
            pp.savefig()
            pp.close()
            plt.close()

    # Plot regulation map FROM each gene
    elif REG_TYPE == 'source':
        reg_source = [x.split('->')[0] for x in reg_name_list]
        source_name = list(sorted(set(reg_source)))

        # Add directory if not exist
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Plot dynamic regulation map (only when less than 10 genes)
        for i in range(len(source_name)):
            print(source_name[i])
            source_index = [j for j, x in enumerate(reg_source) if x == source_name[i]]
            dyn_reg_map_source = dyn_reg_map.iloc[source_index, :]

            plt.figure(figsize=(12, 12))  # This size should be automatically optimized
            seaborn.heatmap(dyn_reg_map_source, cmap="Reds", linewidths=0.01, linecolor='grey', yticklabels=True)
            pp = PdfPages(OUTPUT_DIR + "/" + DATASET_NAME + "_from_" + source_name[i] + "_dynamic_regulation_map.pdf")
            pp.savefig()
            pp.close()
            plt.close()

    else:
        print("ERROR: --type is invalid. designate 'target' or 'source'")
        sys.exit()
