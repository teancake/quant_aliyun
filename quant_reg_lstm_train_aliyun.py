import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import torch
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta
import torch.nn as nn

from lstm_model import get_model, device
import pickle

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'




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

    label_roc = ext["label_roc"].values.astype(float)
    label_close = ext["label_close"].values.astype(float)

    print("#### roc pred {} ... {}\n roc label {} ... {}\n".format(label_roc_pred[0:10], label_roc_pred[-10:],
                                                                                  label_roc[0:10], label_roc[-10:]))
    print("#### roc mse {}, close mse {}".format(mean_squared_error(label_roc, label_roc_pred),
                                              mean_squared_error(label_close, label_close_pred)))

    pred = label_roc_pred
    label = label_roc
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
    num_layers = args.n_layers
    dropout = args.drop_prob
    hidden_size = args.hidden_dim
    epoch_num = args.epoch_num
    data_file_name = args.data_file_name
    use_roc_label = args.use_roc_label


    model_config = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "sequence_length": 5
    }

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


    input_size = train_data_x.shape[-1]
    output_size = train_data_y.shape[-1]
    model_config["input_size"] = input_size
    model_config["output_size"] = output_size
    model = get_model("ae_lstm", config=model_config)

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
            outputs = model.get_outputs(inputs)
            # print("model outputs shape {}".format(outputs.shape))
            # print("inputs {}, targets {}, outputs {}".format(inputs, targets, outputs))
            # loss = loss_fn(outputs[:,-1], targets[:,-1])
            loss = model.get_loss(outputs, targets)
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
                vloss = model.get_loss(voutputs, vlabels)
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
    parser.add_argument('--use_roc_label', type=int, default=1, help='use roc label, 1 true, 0 false')


    args = parser.parse_args()
    print("arguments {}".format(args))

    train(args)


