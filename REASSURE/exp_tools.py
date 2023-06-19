import torch, numpy as np, pathlib, os, random
from PIL import Image


def find_buggy_inputs(dataloader, model, repair_num):
    """For a neural network, we find it's buggy input from a dataset as well as a cut point,
     to seperate the test set."""
    model.eval()
    buggy_inputs, right_label = [], []
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            buggy_index = [i for i in range(len(X)) if pred[i].argmax() != y[i]]
            buggy_inputs += X[buggy_index]
            right_label += y[buggy_index]
            if len(buggy_inputs) >= repair_num:
                buggy_inputs, right_label = buggy_inputs[:repair_num], right_label[:repair_num]
                break
    if len(buggy_inputs) < repair_num:
        print('Not find enough buggy points!')
    return torch.stack(buggy_inputs), torch.stack(right_label)


def specification_matrix_from_labels(right_labels, dim=10):
    """For a classification problem, we translate the specification of 'class should be i' to matrix constraints."""
    P_list, qu_list, ql_list = [], [], []
    for k in right_labels:
        e = np.zeros([1, dim])
        e[0][k] = 1
        P = np.eye(dim) - np.matmul(np.ones([dim, 1]), e)
        P_list.append(np.delete(P, k, 0))
        qu_list.append(np.zeros(dim-1))
        ql_list.append(np.ones(dim-1) * -100)
    return P_list, ql_list, qu_list


def test_diff_on_dataloader(dataloader, model1, model2):
    """Compute the average/maximum norm difference between two neural network"""
    total = 0
    diff_inf_sum, diff_2_sum = torch.tensor(0.0), torch.tensor(0.0)
    with torch.no_grad():
        for X, _ in dataloader:
            pred1, pred2 = model1(X.float()), model2(X.float())
            diff = torch.softmax(pred1, dim=-1) - torch.softmax(pred2, dim=-1)
            diff_inf = torch.norm(diff, dim=-1, p=np.inf)
            diff_2 = torch.norm(diff, dim=-1, p=2)
            total += len(X)
            diff_inf_sum += diff_inf.sum()
            diff_2_sum += diff_2.sum()
    return diff_inf_sum/total, diff_2_sum/total


def success_repair_rate(model, buggy_inputs, right_label, is_print=False):
    with torch.no_grad():
        pred = model(buggy_inputs)
        correct = (pred.argmax(1) == right_label).type(torch.float).sum().item()
    if is_print:
        print('success repair rate:', correct/len(buggy_inputs))
    return correct/len(buggy_inputs)


def test_acc(dataloader, model):
    correct, total_num = 0, 0
    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            if len(y.size()) > 1:
                y = y.argmax(1)
            total_num += len(X)
            # X, y = X.to(device), y.to(device)
            pred = model(X.float())
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    return correct/total_num
