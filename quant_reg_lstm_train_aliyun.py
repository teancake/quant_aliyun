import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import torch
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta
import torch.nn as nn

import pickle

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_dim=512, n_layers=1, drop_prob=0.0):
        super(LSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        # self.bn_in = nn.BatchNorm1d(sequence_length, affine=False)
        # self.bn_out = nn.BatchNorm1d(hidden_dim)


        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        # self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        # print("x size {}".format(x.shape))
        batch_size = x.size(0)
        # print("x before bn {}".format(x))
        # x = self.bn_in(x)
        # print("x after bn {}".format(x))
        lstm_out, hidden = self.lstm(x, hidden)
        # print("lstm_out size {}, hidden 0 size {}, hidden 1 size {}".format(lstm_out.shape, hidden[0].shape, hidden[1].shape))

        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        # lstm_out = self.bn_out(lstm_out)
        # print("lstm_out before drop out {}".format(lstm_out.shape))
        # out = self.dropout(lstm_out)
        # print("out before fc {}".format(out.shape))
        out = self.fc(lstm_out)
        # print("out after fc {}".format(out.shape))

        # out = out.view(batch_size, -1)
        # out = out[:, -1]
        # print("out after -1 {}".format(out.shape))
        return out, hidden

    def init_hidden(self, batch_size, method="zero"):
        weight = next(self.parameters()).data
        if method == "zero":
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).normal_(0, 0.01).to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).normal_(0, 0.01).to(device))

        return hidden




def load_data_from_file(file_path):
    file = open(file_path, "rb")
    data = pickle.load(file)
    file.close()
    return data



def compute_precision_recall(ext, score, use_roc_label):
    th = 5
    label_th = 2

    close = ext["close"].values.astype(float)
    if use_roc_label:
        label_roc_pred = score / 100
        label_close_pred = np.add(close, np.multiply(close, label_roc_pred))
    else:
        label_close_pred = score
        label_roc_pred = np.divide(np.subtract(score, close), close) * 100

    ext["label_roc_pred"] = label_roc_pred
    ext["label_close_pred"] = label_close_pred

    pred = ext["label_roc_pred"].values.astype(float)
    label = ext["label_roc"].values.astype(float)
    print("pred {} ... {}, label {} ... {}, mse {}".format(pred[0:10], pred[-10:], label[0:10], label[-10:],
                                                           mean_squared_error(label, pred)))
    tp = len(label[np.logical_and(label > label_th, pred > th)])
    pp = len(pred[pred > th])
    p = len(label[label > th])

    precision = tp / pp if pp > 0 else 0
    recall = tp / p if p > 0 else 0
    p_ratio = p/len(label)
    pp_ratio = pp / len(pred)
    print("roc threshold {}, precision {}, recall {}, p ratio {}, pp ratio {}".format(th, precision, recall, p_ratio,
                                                                                      pp_ratio))



def train(args):
    batch_size = args.batch_size
    learning_rate = args.lr
    n_layers = args.n_layers
    drop_prob = args.drop_prob
    hidden_dim = args.hidden_dim
    epoch_num = args.epoch_num
    data_file_name = args.data_file_name
    use_roc_label = args.use_roc_label

    sequential_data = load_data_from_file("/mnt/data/{}".format(data_file_name))
    train_data_x, train_data_y, test_data_x, test_data_y, test_data_ext, pred_data_x, pred_data_ext = sequential_data

    print("data loaded")
    print("train x shape {}, train y shape {}, train y mean {}, variance {}".format(train_data_x.shape, train_data_y.shape, np.mean(train_data_y), np.var(train_data_y)))
    print("test x shape {}, test y shape {}, test y mean {}, variance {}".format(test_data_x.shape, test_data_y.shape, np.mean(test_data_y), np.var(test_data_y)))
    print("pred x shape {}, append x shape {}".format(pred_data_x.shape, pred_data_ext.shape))

    train_data_x = torch.tensor(train_data_x, dtype=torch.float32)
    train_data_y = torch.tensor(train_data_y, dtype=torch.float32)
    test_data_x = torch.tensor(test_data_x, dtype=torch.float32)
    test_data_y = torch.tensor(test_data_y, dtype=torch.float32)
    pred_data_x = torch.tensor(pred_data_x, dtype=torch.float32)
    train_dataset = TensorDataset(train_data_x, train_data_y)
    val_dataset = TensorDataset(test_data_x, test_data_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)

    model = LSTM(input_size=train_data_x.shape[-1], output_size=train_data_y.shape[-1], hidden_dim=hidden_dim, n_layers=n_layers, drop_prob=drop_prob)
    model.to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("model created {}".format(model))

    # Run the training loop
    h = model.init_hidden(batch_size, method="normal")
    metric = []
    for epoch in range(0, epoch_num):
        print("Starting epoch {}".format(epoch + 1))

        train_losses = []
        model.train(True)
        for batch_num, data in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            # print("input shape {}, target shape {}, h0 shape {}, h1 shape {}".format(inputs.shape, targets.shape, h[0].shape, h[1].shape))
            h = tuple([each.data for each in h])
            outputs, _ = model(inputs, h)
            # print("model outputs shape {}".format(outputs.shape))
            # print("inputs {}, targets {}, outputs {}".format(inputs, targets, outputs))
            # loss = loss_fn(outputs[:,-1], targets[:,-1])
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if batch_num % 5 == 0:
                print("Loss after batch {}: {}".format(batch_num, np.mean(train_losses)))
        print("Training process in epoch {} has finished. Evaluation started.".format(epoch + 1))

        val_losses = []
        val_labels = []
        val_outputs = []
        model.eval()
        with torch.no_grad():
            val_h = model.init_hidden(batch_size)
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                val_h = tuple([each.data for each in val_h])
                voutputs, _ = model(vinputs, val_h)
                # print("voutputs shape {}, vlabels shape {}".format(voutputs.shape, vlabels.shape))
                # only the last day matters
                # vloss = loss_fn(voutputs[:,-1], vlabels[:,-1])
                # all predictions matters
                vloss = loss_fn(voutputs, vlabels)
                val_labels.extend(vlabels[:, -1].squeeze().detach().cpu().numpy())
                val_outputs.extend(voutputs[:, -1].squeeze().detach().cpu().numpy())
                # print("vloss {}, mse {}".format(vloss, mean_squared_error(val_outputs, val_labels)))
                val_losses.append(vloss.item())
        print("epoch {} train loss {}, val loss {}".format(epoch + 1, np.mean(train_losses), np.mean(val_losses)))
        print("val_labels {} ... {}".format(val_labels[0:10], val_labels[-10:]))
        print("val_outputs {} ... {}".format(val_outputs[0:10], val_outputs[-10:]))
        metric.append([epoch + 1, np.mean(train_losses), np.mean(val_losses)])
    print("metrics {}".format(metric))
    model.eval()
    with torch.no_grad():
        pred, _ = model(test_data_x.to(device), model.init_hidden(test_data_x.shape[0]))
        score = pred[:, -1].squeeze().detach().cpu().numpy()
    compute_precision_recall(test_data_ext, score, use_roc_label)
    pd.set_option('display.max_columns', 20)
    print(test_data_ext.head(10))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch_num', type=int, default=5, help='epoch num')
    parser.add_argument('--batch_size', type=int, default=1001, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--n_layers', type=int, default=1, help='lstm number of layers')
    parser.add_argument('--drop_prob', type=float, default=0.2, help='lstm dropout probability')
    parser.add_argument('--hidden_dim', type=int, default=512, help='lstm hidden variable dimension')
    parser.add_argument('--data_file_name', type=str, default="quant_reg_sequential_data.pkl", help='data file name')
    parser.add_argument('--use_roc_label', type=bool, default=True, help='use roc label')


    args = parser.parse_args()
    print("arguments {}".format(args))

    train(args)


