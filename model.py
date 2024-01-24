import itertools
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, \
    pad_packed_sequence


class Regressor(nn.Module):
    """
    回归器，用于预测时间
    """

    def __init__(self, dim: int, dropout=0.1):
        super().__init__()

        self.linears = nn.ModuleList(
            [
                nn.Linear(dim, dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim // 2, dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim // 4, dim // 8),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim // 8, 1),
            ]
        )

    def forward(self, x):
        for layer in self.linears:
            x = layer(x)
        return x


class Classifier(nn.Module):
    """
    分类器，用于分类
    """

    def __init__(self, dim: int, c: int, dropout=0.1):
        super().__init__()

        self.linears = nn.ModuleList(
            [
                nn.Linear(dim, dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim // 2, dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim // 4, dim // 8),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim // 8, c),
                nn.Softmax(-1),
            ]
        )

    def forward(self, x):
        for layer in self.linears:
            x = layer(x)
        return x


class Wide(nn.Module):
    def __init__(self, wide_config):
        super(Wide, self).__init__()
        self.config = wide_config
        self.linear = nn.Linear(in_features=self.feature_dim,
                                out_features=self.config["out_dim"])

    @property
    def feature_dim(self) -> int:
        dense_dim = self.config["dense"]["size"]
        sparse_dim = sum([feature["size"]
                          for feature in self.config["sparse"]])
        cross_dim = sum(
            [
                feature1["size"] * feature2["size"]
                for feature1, feature2 in itertools.combinations(
                self.config["sparse"], 2)
            ]
        )
        return dense_dim + sparse_dim + cross_dim

    def forward(self, dense, sparse):
        sparse_features = []

        for feature in self.config["sparse"]:
            onehot = F.one_hot(sparse[..., feature["col"]], feature["size"])
            sparse_features.append(onehot)

        for feature1, feature2 in itertools.combinations(self.config["sparse"], 2):
            cross = (
                    sparse[..., feature1["col"]] * feature2["size"]
                    + sparse[..., feature2["col"]]
            )

            onehot = F.one_hot(cross, feature1["size"] * feature2["size"])
            sparse_features.append(onehot)

        sparse_feature = torch.cat(sparse_features, -1)

        features = torch.cat([dense, sparse_feature], -1)

        out = self.linear(features)
        out = F.relu(out)

        return out


class Deep(nn.Module):
    def __init__(self, deep_config, dropout=0.1):
        super(Deep, self).__init__()
        self.config = deep_config

        for feature in self.config["sparse"]:
            setattr(
                self,
                f'embedding_{feature["name"]}',
                nn.Embedding(feature["size"], feature["dim"]),
            )

        h_dim = self.config["out_dim"] * 2
        self.linear1 = nn.Linear(self.feature_dim, h_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(h_dim, h_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(h_dim, self.config["out_dim"])

    @property
    def feature_dim(self) -> int:
        dim = self.config["dense"]["size"]
        for feature in self.config["sparse"]:
            dim += feature["dim"]
        return dim

    def forward(self, dense, sparse):
        sparse_features = []

        for feature in self.config["sparse"]:
            embed = getattr(self, f'embedding_{feature["name"]}')(
                sparse[..., feature["col"]]
            )
            sparse_features.append(embed)

        sparse_feature = torch.cat(sparse_features, -1)

        features = torch.cat([dense, sparse_feature], -1)

        out = self.linear1(features)
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.linear3(out)
        out = F.relu(out)
        return out


class Recurrent(nn.Module):
    def __init__(
            self,
            rnn_config,
            dropout=0.1,
    ):
        super().__init__()
        self.config = rnn_config

        for feature in self.config["sparse"]:
            setattr(
                self,
                f'embedding_{feature["name"]}',
                nn.Embedding(feature["size"], feature["dim"]),
            )

        input_size = self.config["input_size"]
        hidden_size = self.config["hidden_size"]
        num_layers = self.config["num_layers"]
        self.linear_link = nn.Linear(self.feature_dim, input_size)

        self.lstm_link = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False,
        )

        # self.gru_link = nn.GRU(
        #     input_size=input_size,
        #     hidden_size=hidden_size,
        #     num_layers=num_layers,
        #     dropout=dropout,
        #     batch_first=True,
        # )

    @property
    def feature_dim(self) -> int:
        dim = self.config["dense"]["size"]
        for feature in self.config["sparse"]:
            dim += feature["dim"]
        return dim

    def init_hidden_lstm(self, batch_size):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        hidden_size = self.config["hidden_size"]
        num_layers = self.config["num_layers"]
        num_directions = self.config["num_directions"]
        return (torch.zeros(num_layers * num_directions, batch_size, hidden_size, device=device),
                torch.zeros(num_layers * num_directions, batch_size, hidden_size, device=device))

    def init_hidden_gru(self, batch_size):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        hidden_size = self.config["hidden_size"]
        num_layers = self.config["num_layers"]
        return torch.zeros(num_layers, batch_size, hidden_size, device=device)

    def forward(self, seq_dense, seq_sparse, link_len):
        sparse_features = []

        for feature in self.config["sparse"]:
            embed = getattr(self, f'embedding_{feature["name"]}')(
                seq_sparse[..., feature["col"]]
            )
            sparse_features.append(embed)

        sparse_feature = torch.cat(sparse_features, -1)

        seq = torch.cat([seq_dense, sparse_feature], dim=-1)

        seq = self.linear_link(seq)
        seq = F.relu(seq)

        packed_link = pack_padded_sequence(
            seq, link_len, batch_first=True, enforce_sorted=False
        )

        self.begin_state_lstm = self.init_hidden_lstm(len(link_len))
        # self.begin_state_gru = self.init_hidden_gru(len(link_len))

        self.lstm_link.flatten_parameters()

        out_link_lstm, (hn_link_lstm, cn_link_lstm) = self.lstm_link(packed_link, self.begin_state_lstm)
        # out_link_gru, hn_link_gru = self.gru_link(packed_link, self.begin_state_gru)

        out_link_lstm, index = pad_packed_sequence(out_link_lstm, batch_first=True)
        return out_link_lstm
        # return hn_link_lstm[-1]
        # return hn_link_gru[-1]


class WDR(nn.Module):
    def __init__(
            self,
            wide_config,
            deep_config,
            rnn_config,
            dropout=0.1,
    ):
        super(WDR, self).__init__()
        self.wide = Wide(wide_config)
        self.deep = Deep(deep_config, dropout=dropout)
        self.recurrent = Recurrent(rnn_config, dropout=dropout)
        total_dim = wide_config["out_dim"] + deep_config["out_dim"] + \
                    rnn_config["hidden_size"]
        self.regressor = Regressor(total_dim)

    def forward(
            self,
            dense,
            sparse,
            seq_dense,
            seq_sparse,
            seq_len,
    ):
        out_wide = self.wide(dense, sparse)
        out_deep = self.deep(dense, sparse)
        out_recurrent = self.recurrent(seq_dense, seq_sparse, seq_len)
        out_recurrent = torch.stack([out_recurrent[i, seq_len[i] - 1, :]
                                     for i in range(len(out_recurrent))])
        features = torch.cat([out_wide, out_deep, out_recurrent], -1)

        out = self.regressor(features)
        return out


class PTTE(nn.Module):
    def __init__(
            self,
            wide_config,
            deep_config,
            rnn_config,
            dropout=0.1,
    ):
        super(PTTE, self).__init__()
        self.wide = Wide(wide_config)
        self.deep = Deep(deep_config, dropout=dropout)
        self.recurrent = Recurrent(rnn_config, dropout=dropout)
        total_dim = wide_config["out_dim"] + deep_config["out_dim"] + \
                    rnn_config["hidden_size"]
        self.regressor = Regressor(total_dim)
        self.classifier = Classifier(total_dim, 5)

    def forward(
            self,
            dense,
            sparse,
            seq_dense,
            seq_sparse,
            seq_len,
    ):
        out_wide = self.wide(dense, sparse)
        out_deep = self.deep(dense, sparse)
        out_recurrent = self.recurrent(seq_dense, seq_sparse, seq_len)
        out_recurrent = torch.stack([out_recurrent[i, seq_len[i] - 1, :]
                                     for i in range(len(out_recurrent))])
        features = torch.cat([out_wide, out_deep, out_recurrent], -1)

        out1 = self.regressor(features)
        out2 = self.classifier(features)
        return out1, out2


if __name__ == '__main__':
    pass
