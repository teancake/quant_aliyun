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



def compute_precision_recall_updated(label, pred):
    print("#### label mse {}".format(mean_squared_error(label, pred)))

    auc_curve = []
    for th in range(-10, 10):
        label_th = th
        tp = len(label[np.logical_and(label >= label_th, pred >= th)])
        pp = len(pred[pred >= th])
        p = len(label[label >= th])
        n = len(label[label < th])
        fp = len(label[np.logical_and(label < label_th, pred >= th)])

        precision = tp / pp if pp > 0 else 0
        recall = tp / p if p > 0 else 0
        p_ratio = p/len(label)
        pp_ratio = pp / len(pred)
        tpr = recall
        fpr = fp / n if n > 0 else 0
        loss_p = len(label[np.logical_and(label < -label_th, pred >= th)]) / pp if pp > 0 else 0
        print("threshold {}, precision {}, loss probability {}, recall {}, p ratio {}, pp ratio {}, pred len {}, label len {}, p cnt {}, pp cnt {}".format(th, precision, loss_p, recall, p_ratio,
                                                                                      pp_ratio, len(pred), len(label), p, pp))
        auc_curve.append([th, precision, recall, tpr, fpr, pp_ratio])
    auc_curve = np.array(auc_curve)
    # print(auc_curve)

def train(args):
    batch_size = args.batch_size
    learning_rate = args.lr
    num_layers = args.n_layers
    dropout = args.drop_prob
    hidden_size = args.hidden_dim
    epoch_num = args.epoch_num
    data_file_name = args.data_file_name
    use_roc_label = args.use_roc_label
    model_name = args.model_name


    model_config = {
        "model_name": model_name,
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
    print("model config {}".format(model_config))
    model = get_model(model_config)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("model created {}".format(model))

    # Run the training loop
    metric = []
    for epoch in range(0, epoch_num):
        print("Starting epoch {}".format(epoch + 1))

        train_losses = []
        model.train(True)
        for batch_num, data in enumerate(train_loader):

            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            # print("input shape {}, target shape {}, h0 shape {}, h1 shape {}".format(inputs.shape, targets.shape, h[0].shape, h[1].shape))
            loss = model.get_loss(inputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if batch_num + 1 % 50 == 0:
                print("Loss after batch {}: {}".format(batch_num + 1, np.mean(train_losses)))
        print("Training process in epoch {} has finished. Evaluation started.".format(epoch + 1))

        val_losses = []
        val_labels = []
        val_outputs = []
        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                voutputs = model.get_outputs(vinputs)
                # print("voutputs shape {}, vlabels shape {}".format(voutputs.shape, vlabels.shape))
                # only the last day matters
                # vloss = loss_fn(voutputs[:,-1], vlabels[:,-1])
                # all predictions matters
                vloss = model.get_loss(vinputs, vlabels)
                val_labels.extend(vlabels[:, -1].squeeze().detach().cpu().numpy())
                val_outputs.extend(voutputs[:, -1].squeeze().detach().cpu().numpy())
                # print("vloss {}, mse {}".format(vloss, mean_squared_error(val_outputs, val_labels)))
                val_losses.append(vloss.item())
        print("epoch {} train loss {}, val loss {}".format(epoch + 1, np.mean(train_losses), np.mean(val_losses)))
        print("val_labels {} ... {}".format(val_labels[0:10], val_labels[-10:]))
        print("val_outputs {} ... {}".format(val_outputs[0:10], val_outputs[-10:]))
        metric.append([epoch + 1, np.mean(train_losses), np.mean(val_losses)])
    print("metrics {}".format(metric))


    print("validation on the latest model parameters.")
    model.eval()
    with torch.no_grad():
        pred = model.get_outputs(test_data_x.to(device))
        pred = pred[:, -1].squeeze().detach().cpu().numpy()


    label_name = "label_roc"
    label_name_pred = "{}_pred".format(label_name)
    test_data_ext[label_name_pred] = pred
    label = test_data_ext[label_name].values.astype(float)
    compute_precision_recall_updated(label, pred)

    pd.set_option('display.max_columns', 20)
    print(test_data_ext.head(10))
    # print(test_data_y[0:10])


    print("now make predictions")
    model.eval()
    with torch.no_grad():
        pred = model.get_outputs(pred_data_x.to(device))
        pred = pred[:, -1].squeeze().detach().cpu().numpy()

    # quant_data_util.fill_ext_with_predictions(pred_data_ext, pred, use_roc_label)
    pred_data_ext[label_name_pred] = pred
    print("now only use the latest date and find the ups")
    pred_data_ext = pred_data_ext[pred_data_ext["日期"] == pred_data_ext["日期"].max()]
    pred_data_ext.sort_values(by=[label_name_pred], inplace=True, ascending=False)

    print(pred_data_ext.head(20).to_string())

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
    parser.add_argument('--model_name', type=str, default="lstm", help='model name')



    args = parser.parse_args()
    print("arguments {}".format(args))

    train(args)


