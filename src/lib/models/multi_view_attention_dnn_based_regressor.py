##########################################
# Project: Multi-View Attention DNN-based regression model
# Author: Yusuke Hiki
##########################################

# Modules
import torch


# Attention-DNN based Regressor
class MultiViewAttentionDNNBasedRegressor(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        #self.encoder1 = torch.nn.Linear(in_features=input_size, out_features=hidden_size)
        #self.encoder2 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_size, out_features=hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_size, out_features=hidden_size),
            torch.nn.ReLU(),
        )
        self.fc = torch.nn.Linear(in_features=hidden_size, out_features=1)

    def pretrain(self, x):
        # Get number of timepoint
        n_timepoint = x.shape[1]

        # Get feature vector of all genes expresesion outputted from LSTM
        x_all_size = [1, x.shape[1], x.shape[0]]
        x_all = torch.reshape(torch.transpose(x, 0, 1), x_all_size)
        h_all = self.encoder(x_all)

        # Get regression result without attention
        y_pred = torch.reshape(self.fc(h_all), [n_timepoint])

        return y_pred

    def forward(self, x):
        # Initialize feature vector h_capital and attention_weight
        n_gene, n_timepoint = x.shape[0], x.shape[1]

        # Get feature vector of all genes expresesion outputted from DNN
        x_all_size = [1, x.shape[1], x.shape[0]]
        x_all = torch.reshape(torch.transpose(x, 0, 1), x_all_size)

        xi_size = [1, x.shape[1], x.shape[0]]
        x_in = torch.reshape(torch.transpose(x, 0, 1), xi_size)

        for i_gene in range(n_gene):
            # Get expression with target gene (other gene expression is masked)
            xi = torch.clone(x)
            i_mask = list(set(range(n_gene)) - set([i_gene]))
            xi[i_mask, :] = 0
            xi = torch.reshape(torch.transpose(xi, 0, 1), xi_size)
            x_in = torch.cat([x_in, xi], dim=0)

        h = self.encoder(x_in)

        # Get attention weight
        attention_weight = torch.matmul(torch.transpose(h[1:], 0, 1), torch.transpose(torch.transpose(h[0:1], 0, 1), 1, 2))
        attention_weight = torch.nn.functional.softmax(input=attention_weight, dim=1)

        # Regression with attention
        h_attention = torch.matmul(torch.transpose(attention_weight, 1, 2), torch.transpose(h[1:], 0, 1))
        y_pred = torch.transpose(self.fc(h_attention), 0, 2)[0, 0]

        # Calculate importance score from attention weight
        attention_weight = torch.transpose(attention_weight, 0, 2)[0]
        imp_score = torch.sum(attention_weight, dim=1)

        '''
        y_pred: Regression result
        attention_weight: time- and gene-view score for regulation to target gene
        imp_score: importance score of each gene for regulation to target gene 
        '''
        return y_pred, attention_weight, imp_score

