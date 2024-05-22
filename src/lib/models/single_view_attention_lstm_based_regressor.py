##########################################
# Project: Single-View Attention LSTM-based regression model
# Author: Yusuke Hiki
##########################################

# Modules
import torch


# Attention-LSTM based Regressor
class SingleViewAttentionLSTMBasedRegressor(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.encoder = torch.nn.LSTM(input_size=input_size,
                                     hidden_size=hidden_size,
                                     num_layers=num_layers,
                                     bidirectional=False)
        self.fc = torch.nn.Linear(in_features=hidden_size, out_features=1)

    def pretrain(self, x):
        # Get number of timepoint
        n_timepoint = x.shape[1]

        # Get feature vector of all genes expresesion outputted from LSTM
        x_all_size = [1, x.shape[1], x.shape[0]]
        x_all = torch.reshape(torch.transpose(x, 0, 1), x_all_size)
        h_all, _ = self.encoder(x_all)

        # Get regression result without attention
        y_pred = torch.reshape(self.fc(h_all), [n_timepoint])

        return y_pred

    def forward(self, x):
        # Initialize feature vector h_capital and attention_weight
        n_gene, n_timepoint = x.shape[0], x.shape[1]

        # Get feature vector of all genes expresesion outputted from LSTM
        x_all_gene_size = [1, x.shape[1], x.shape[0]]
        x_all_gene = torch.reshape(torch.transpose(x, 0, 1), x_all_gene_size)
        h_all_gene, _ = self.encoder(x_all_gene)

        # Get regression result based on expression of each gene
        for i_gene in range(n_gene):
            # Get expression with target gene (other gene expression is masked)
            xi = torch.clone(x)
            i_mask = list(set(range(n_gene)) - set([i_gene]))
            xi[i_mask, :] = 0
            xi_size = [1, x.shape[1], x.shape[0]]
            xi = torch.reshape(torch.transpose(xi, 0, 1), xi_size)

            # Get feature vector of each gene expression outputted from LSTM
            hi, _ = self.encoder(xi)

            if i_gene == 0:
                # yi_pred = torch.hstack([yi_pred, torch.reshape(self.fc(hi), [1, n_timepoint])])
                yi_pred = torch.reshape(self.fc(hi), [1, n_timepoint])
            else:
                # yi_pred = torch.vstack([yi_pred, torch.reshape(self.fc(hi), [1, n_timepoint])])
                yi_pred = torch.cat([yi_pred, torch.reshape(self.fc(hi), [1, n_timepoint])], dim=0)

        # Get regression result using FC
        y_pred_all_gene = torch.reshape(self.fc(h_all_gene), [1, n_timepoint])

        # Calculate attention weight for each gene contribution to regression based on all genes expression
        attention_weight = torch.matmul(y_pred_all_gene, torch.transpose(yi_pred, 0, 1))
        attention_weight = torch.nn.functional.softmax(attention_weight, dim=1)

        y_pred = torch.matmul(attention_weight, yi_pred)[0]

        return y_pred, yi_pred, attention_weight[0]


class SingleViewParameterizedAttentionLSTMBasedRegressor(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.encoder = torch.nn.LSTM(input_size=input_size,
                                     hidden_size=hidden_size,
                                     num_layers=num_layers,
                                     bidirectional=False)
        self.fc = torch.nn.Linear(in_features=hidden_size, out_features=1)
        self.attention_fc = torch.nn.Linear(in_features=input_size, out_features=1, bias=False)

    def pretrain(self, x):
        # Get number of timepoint
        n_timepoint = x.shape[1]

        # Get feature vector of all genes expresesion outputted from LSTM
        x_all_size = [1, x.shape[1], x.shape[0]]
        x_all = torch.reshape(torch.transpose(x, 0, 1), x_all_size)
        h_all, _ = self.encoder(x_all)

        # Get regression result without attention
        y_pred = torch.reshape(self.fc(h_all), [n_timepoint])

        return y_pred

    def forward(self, x):
        # Initialize feature vector h_capital and attention_weight
        n_gene, n_timepoint = x.shape[0], x.shape[1]

        # Get regression result based on expression of each gene
        for i_gene in range(n_gene):
            # Get expression with target gene (other gene expression is masked)
            xi = torch.clone(x)
            i_mask = list(set(range(n_gene)) - set([i_gene]))
            xi[i_mask, :] = 0
            xi_size = [1, x.shape[1], x.shape[0]]
            xi = torch.reshape(torch.transpose(xi, 0, 1), xi_size)

            # Get feature vector of each gene expression outputted from LSTM
            hi, _ = self.encoder(xi)

            if i_gene == 0:
                # yi_pred = torch.hstack([yi_pred, torch.reshape(self.fc(hi), [1, n_timepoint])])
                yi_pred = torch.reshape(self.fc(hi), [1, n_timepoint])
            else:
                # yi_pred = torch.vstack([yi_pred, torch.reshape(self.fc(hi), [1, n_timepoint])])
                yi_pred = torch.cat([yi_pred, torch.reshape(self.fc(hi), [1, n_timepoint])], dim=0)

        y_pred = torch.transpose(self.attention_fc(torch.transpose(yi_pred, 0, 1)), 0, 1)[0]
        # attention_weight = self.attention_fc.weight.data
        attention_weight = torch.nn.functional.softmax(torch.abs(self.attention_fc.weight.data), dim=1)

        return y_pred, yi_pred, attention_weight[0]
