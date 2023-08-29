import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta
import torch.nn as nn

import pickle

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_dim=512, n_layers=1, drop_prob=0.0, sequence_length=1):
        super(LSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        # self.bn_in = nn.BatchNorm1d(sequence_length)
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

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden



def load_data_from_file(file_path):
    file = open(file_path, "rb")
    data = pickle.load(file)
    file.close()
    return data




def train(args):
    sequence_length = 15
    batch_size = args.batch_size
    learning_rate = args.lr
    n_layers = args.n_layers
    drop_prob = args.drop_prob
    hidden_dim = args.hidden_dim
    epoch_num = args.epoch_num
    data_file_name = args.data_file_name

    sequential_data = load_data_from_file("/mnt/data/{}".format(data_file_name))
    train_data_x, train_data_y, test_data_x, test_data_y, test_data_ext, pred_data_x, pred_data_ext = sequential_data

    pred_data_x = pred_data_x[0:2,:,:]
    pred_data_ext = pred_data_ext.head(2)
    print("train x shape {}, train y shape {}, train y mean {}, variance {}".format(train_data_x.shape, train_data_y.shape, np.mean(train_data_y), np.var(train_data_y)))
    print("test x shape {}, test y shape {}, test y mean {}, variance {}".format(test_data_x.shape, test_data_y.shape, np.mean(test_data_y), np.var(test_data_y)))
    print("pred x shape {}, append x shape {}".format(pred_data_x.shape, pred_data_ext.shape))

    print("data loaded")
    train_data_x = torch.tensor(train_data_x, dtype=torch.float32)
    train_data_y = torch.tensor(train_data_y, dtype=torch.float32)
    test_data_x = torch.tensor(test_data_x, dtype=torch.float32)
    test_data_y = torch.tensor(test_data_y, dtype=torch.float32)
    pred_data_x = torch.tensor(pred_data_x, dtype=torch.float32)
    train_dataset = TensorDataset(train_data_x, train_data_y)
    val_dataset = TensorDataset(test_data_x, test_data_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)

    model = LSTM(input_size=train_data_x.shape[-1], n_layers=n_layers, drop_prob=drop_prob, sequence_length=sequence_length, hidden_dim=hidden_dim)
    model.to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("model created {}".format(model))

    # Run the training loop
    h = model.init_hidden(batch_size)

    for epoch in range(0, epoch_num):
        print("Starting epoch {}".format(epoch + 1))

        train_losses = []
        model.train(True)
        for batch_num, data in enumerate(train_loader):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            # print("input shape {}, target shape {}, h0 shape {}, h1 shape {}".format(inputs.shape, targets.shape, h[0].shape, h[1].shape))
            h = tuple([each.data for each in h])
            outputs, _ = model(inputs, h)
            # print("model outputs shape {}".format(outputs.shape))
            loss = loss_fn(outputs, targets)

            # 2. backward
            optimizer.zero_grad()  # reset gradient
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if batch_num % 5 == 0:
                print("Loss after batch {}: {}".format(batch_num, np.mean(train_losses)))
        print("Training process in epoch {} has finished. Evaluation started.".format(epoch + 1))

        val_losses = []
        model.eval()
        with torch.no_grad():
            val_h = model.init_hidden(batch_size)
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                val_h = tuple([each.data for each in val_h])
                voutputs, val_h = model(vinputs, val_h)
                vloss = loss_fn(voutputs, vlabels)
                val_losses.append(vloss.item())
        print("epoch {} train loss {}, val loss {}".format(epoch + 1, np.mean(train_losses), np.mean(val_losses)))
    #
    # run_id = uuid.uuid1().hex
    # model_name = "lstm"
    # print("saving validation results")
    # test_data_x = test_data_x.to(device)
    # pred, _ = model(test_data_x, model.init_hidden(test_data_x.shape[0]))
    # test_data_ext["score"] = pred.squeeze().cpu().detach().numpy()
    #
    # print("now make predictions")
    # pred_data_x = pred_data_x.to(device)
    # pred_h = model.init_hidden(pred_data_x.shape[0])
    # pred, _ = model(pred_data_x, pred_h)
    # print("pred x {}".format(pred_data_x))
    # print("pred shape {}, pred {}".format(pred.shape, pred))
    # pred_data_ext["score"] = pred.squeeze().cpu().detach().numpy()
    # pred_data_disp = pred_data_ext.copy()
    # print("now find the ups")
    # pred_data_disp.sort_values(by=["score"], inplace=True, ascending=False)
    # print(pred_data_disp.head(20).to_string())
    #
    # print("now make predictions again")
    # pred_data_x = pred_data_x.to(device)
    # pred_h = model.init_hidden(pred_data_x.shape[0])
    # pred, _ = model(pred_data_x, pred_h)
    # print("pred x {}".format(pred_data_x))
    # print("pred shape {}, pred {}".format(pred.shape, pred))
    # pred_data_ext["score"] = pred.squeeze().cpu().detach().numpy()
    # pred_data_disp = pred_data_ext.copy()
    # print("now find the ups")
    # pred_data_disp.sort_values(by=["score"], inplace=True, ascending=False)
    # print(pred_data_disp.head(20).to_string())
    #
    # print("now make predictions for the third time")
    # pred_data_x = pred_data_x.to(device)
    # pred_h = model.init_hidden(pred_data_x.shape[0])
    # pred, _ = model(pred_data_x, pred_h)
    # print("pred x {}".format(pred_data_x))
    # print("pred shape {}, pred {}".format(pred.shape, pred))
    # pred_data_ext["score"] = pred.squeeze().cpu().detach().numpy()
    # pred_data_disp = pred_data_ext.copy()
    # print("now find the ups")
    # pred_data_disp.sort_values(by=["score"], inplace=True, ascending=False)
    # print(pred_data_disp.head(20).to_string())


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

    args = parser.parse_args()
    print("arguments {}".format(args))

    train(args)


