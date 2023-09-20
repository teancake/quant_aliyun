import torch
from abc import ABC, abstractmethod

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import torch
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def get_outputs(self, inputs):
        pass

    @abstractmethod
    def get_loss(self, inputs, targets):
        pass

#
# class AttnLstm(BaseModel):
# https://github.com/LogicJake/Tencent_Ads_Algo_2020_TOP12/blob/master/README.md
#     def __init__(self):
#         super(AttnLstm, self).__init__()
#         self.seq_embedding_features = seq_embedding_features
#         self.statistics_features = statistics_features
#         self.seq_statistics_features = seq_statistics_features
#
#         self.seq_len = seq_len
#
#         self.seq_statistics_size = len(seq_statistics_features)
#         self.statistics_size = len(statistics_features)
#
#         self.device = device
#
#         input_size = 0
#         self.embeds = nn.ModuleDict()
#
#         for f in self.seq_embedding_features:
#             embedding_layer = nn.Embedding(
#                 self.seq_embedding_features[f]['nunique'],
#                 self.seq_embedding_features[f]['embedding_dim'])
#
#             pretrained_weight = np.array(
#                 self.seq_embedding_features[f]['pretrained_embedding'])
#             embedding_layer.weight.data.copy_(
#                 torch.from_numpy(pretrained_weight))
#             embedding_layer.weight.requires_grad = False
#             self.embeds[f] = embedding_layer
#
#         # LSTM 层
#         self.lstm = nn.LSTM(input_size,
#                             128,
#                             batch_first=True,
#                             num_layers=2,
#                             bidirectional=True)
#
#         # Attention 层

#         # categorical feature embedding 层

#
#         # DNN 层
#         dnn_input_size = self.attention_output_size + attention_input_size + self.statistics_size
#
#         self.linears = nn.Sequential(nn.Linear(dnn_input_size, 1024),
#                                      nn.LeakyReLU(), nn.BatchNorm1d(1024),
#                                      nn.Linear(1024, 256), nn.LeakyReLU(),
#                                      nn.BatchNorm1d(256), nn.Linear(256, 64),
#                                      nn.LeakyReLU(), nn.BatchNorm1d(64),
#                                      nn.Linear(64, 16), nn.LeakyReLU(),
#                                      nn.BatchNorm1d(16), nn.Dropout(0.1))
#
#         # age 输出层
#         self.age_output = nn.Linear(16, 10)
#
#     def forward(self, seq_id_list, statistics_input, statistics_seq_input_list,
#                 seq_lengths):
#         batch_size = seq_id_list[0].shape[0]
#
#         # 序列 id Embedding
#         seq_feature_list = []
#         for i, seq_id in enumerate(seq_id_list):
#             feature_name = list(self.seq_embedding_features.keys())[i]
#             embeddings = self.embeds[feature_name](seq_id.to(self.device))
#             seq_feature_list.append(embeddings)
#
#         seq_input = torch.cat(seq_feature_list, 2)
#         seq_input = F.dropout2d(seq_input, 0.1, training=self.training)
#
#         # LSTM
#         seq_output, _ = self.lstm(seq_input)
#         # mask padding
#         mask = torch.zeros(seq_output.shape).to(self.device)
#         for idx, seqlen in enumerate(seq_lengths):
#             mask[idx, :seqlen] = 1
#
#         seq_output = seq_output * mask
#         lstm_output_max, _ = torch.max(seq_output, dim=1)
#
#         # Attention
#         Q = self.Q_weight(seq_output)
#         K = self.K_weight(seq_output)
#         V = self.V_weight(seq_output)
#
#         tmp = torch.bmm(Q, K.transpose(1, 2))
#         tmp = tmp / np.sqrt(self.attention_output_size)
#         w = torch.softmax(tmp, 2)
#         att_output = torch.bmm(w, V)
#         att_output = att_output * mask
#         att_max_output, _ = torch.max(att_output, dim=1)
#
#         # 拼接统计特征
#         cat_output = torch.cat(
#             [att_max_output, lstm_output_max, statistics_input], 1)
#
#         # DNN
#         dnn_output = self.linears(cat_output)
#         age_output = self.age_output(dnn_output)
#
#         return age_output
#
#
# def get_loss(self, inputs, targets):
#         pass
#
#     def get_outputs(self, inputs):
#         pass
#
#

class LSTM(BaseModel):
    def __init__(self, input_size, output_size=1, hidden_size=512, num_layers=1, dropout=0.0):
        super(LSTM, self).__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # self.bn_in = nn.BatchNorm1d(sequence_length, affine=False)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        # print("x size {}".format(x.shape))
        batch_size = x.size(0)
        # print("x before bn {}".format(x))
        # print("x shape {}, before batch norm {}".format(x.shape, x))
        # x = self.bn_in(x)
        # print("x shape {}, after batch norm {}".format(x.shape, x))
        # print("x after bn {}".format(x))

        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)


        # print("lstm_out size {}, hidden 0 size {}, hidden 1 size {}".format(lstm_out.shape, hidden[0].shape, hidden[1].shape))

        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        # lstm_out = self.bn_out(lstm_out)
        # print("lstm_out before drop out {}".format(lstm_out.shape))
        attn_out, attn_weight = self.multihead_attn(lstm_out, lstm_out, lstm_out)
        out = attn_out
        # out = lstm_out
        # print("out before fc {}".format(out.shape))
        out = self.fc(out)
        # out = self.head(out)
        # print("out after fc {}".format(out.shape))
        return out, hidden

    def init_hidden(self, batch_size, method="zero"):
        weight = next(self.parameters()).data
        if method == "zero":
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device),
                      weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device))
        else:
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).normal_(0, 0.01).to(device),
                      weight.new(self.num_layers, batch_size, self.hidden_size).normal_(0, 0.01).to(device))

        return hidden

    def get_outputs(self, inputs):
        outputs, _ = self.forward(inputs)
        return outputs

    def get_loss(self, inputs, targets):
        outputs = self.get_outputs(inputs)
        # print("model outputs shape {}".format(outputs.shape))
        # print("inputs {}, targets {}, outputs {}".format(inputs, targets, outputs))
        # loss = loss_fn(outputs[:,-1], targets[:,-1])
        loss = torch.nn.MSELoss()(outputs, targets)
        return loss





class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).to(device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AeLstm(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(AeLstm, self).__init__()

        self.noise = AddGaussianNoise(std=0.2)

        ae_hidden_size = hidden_size
        lstm_hidden_size = hidden_size

        self.ae_encoder = nn.Linear(input_size, ae_hidden_size)
        self.ae_act = nn.ReLU()
        self.ae_decoder = nn.Linear(ae_hidden_size, input_size)
        self.rnn = nn.LSTM(input_size + ae_hidden_size, lstm_hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout)

        self.dropout = nn.Dropout(p=0.2)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden_size + input_size, lstm_hidden_size),
            nn.LayerNorm(lstm_hidden_size),
            nn.ReLU(),
            nn.Linear(lstm_hidden_size, 1),
        )

    def forward(self, _x, _y=None):
        if self.training:
            h = self.noise(_x)
        else:
            h = _x

        ae_h1 = self.ae_act(self.ae_encoder(h))
        ae_h2 = self.ae_decoder(ae_h1)
        ae_loss = nn.MSELoss()(ae_h2, h)
        # print("ae_h1 {}, ae_h2 {}, ae_loss {}, {}".format(ae_h1.shape, ae_h2.shape, ae_loss.shape, ae_loss))
        # print("_x shapes {}, ae_h1 shapes {}".format(_x.shape, ae_h1.shape))
        h = torch.cat([_x, ae_h1], dim=-1)
        # print("h shapes {}".format(h.shape))

        h, _ = self.rnn(h)
        h = self.dropout(h)
        h = torch.cat([_x, h], dim=-1)
        y = self.head(h)
        # y = y.squeeze(1)
        # print("y shape {}, _y shape {}".format(y.shape, _y.shape))

        if _y is None:
            return y, None

        loss = nn.MSELoss()(y, _y.squeeze(1))
        # print("loss shape {}, ae_loss shape {}".format(loss.shape, ae_loss.shape))

        loss = loss + ae_loss

        return y, loss

    def get_outputs(self, inputs):
        outputs, loss = self.forward(inputs)
        return outputs

    def get_loss(self, inputs, targets):
        outputs, loss = self.forward(inputs, targets)
        return loss


def get_model(config) -> BaseModel:
    model_name = config.get("model_name")
    if model_name == "lstm":
        input_size = config.get("input_size")
        output_size = config.get("output_size")
        hidden_size = config.get("hidden_size", 32)
        num_layers = config.get("num_layers", 1)
        dropout = config.get("dropout", 0.2)
        model = LSTM(input_size=input_size, output_size=output_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    elif model_name == "ae_lstm":
        input_size = config.get("input_size")
        output_size = config.get("output_size")
        hidden_size = config.get("hidden_size", 32)
        num_layers = config.get("num_layers", 1)
        dropout = config.get("dropout", 0.2)
        model = AeLstm(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    else:
        print("model {} not found.".format(model_name))
        model = None
    model.to(device)
    return model

if __name__ == '__main__':
    # lstm = torch.nn.LSTM(input_size=3, hidden_size=1)
    output_size = 2
    lstm = torch.nn.LSTM(3,output_size)
    print(lstm)
    inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

    # initialize the hidden state.
    hidden = (torch.randn(1, 1, output_size),
              torch.randn(1, 1, output_size))
    for i in inputs:
        # Step through the sequence one element at a time.
        # after each step, hidden contains the hidden state.
        out, hidden = lstm(i.view(1, 1, -1), hidden)


    inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    hidden = (torch.randn(1, 1, output_size), torch.randn(1, 1, output_size))  # clean out hidden state
    out, hidden = lstm(inputs, hidden)
    print(out)
    print(hidden)
