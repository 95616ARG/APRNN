import torch, torch.nn as nn, numpy as np, math, time
from REASSURE.tools.gurobi_solver import solve_LP
# from scipy.optimize import linprog
from REASSURE.exp_tools import specification_matrix_from_labels


def linear_region_from_input(x, all_neurons, bounds):
    """
    :param x: input of neural network;
    :param all_neurons: all the hidden neurons of x;
    :param bounds: neural network bounds for all inputs.
    bounds = [lowerbound_list, upperbound_list], lowerbound is a np.array of lowerbound for every dimension;
    :return: H-rep of a linear region, {x| Ax <= b}.
    """
    A_list, b_list = [], []
    for neuron in all_neurons:
        neuron.backward(retain_graph=True)
        grad_x = x.grad
        A = grad_x.clone().detach()
        b = torch.matmul(A, x.squeeze()) - neuron
        if neuron >= 0:
            A, b = -A, -b
        A_list.append(A.detach().numpy())
        b_list.append(b.detach().numpy())
    A = np.concatenate([-np.eye(len(bounds[0])), np.eye(len(bounds[1])), np.stack(A_list)])
    b = np.concatenate([-bounds[0], bounds[1], np.stack(b_list)])
    "-x <= -bounds[0]; x <= bounds[1] <=> "
    "bounds[0] <= x <= bounds[1]"
    return np.float32(A), np.float32(b)


def is_redundant(A, b, c, d, is_gurobi):
    # Ax <= b  ==>  cx <= d is redundant.
    # res = min -cx = - max cx.
    # res >= -d <==>  - max cx >= -d  <==>  max cx <= d <==> constraint i is redundant
    if len(A) == 0:
        return False
    if not is_gurobi:
        print("Forcing gurobi")
        is_gurobi = True
    if is_gurobi:
        res = solve_LP(c, A, b, bounds=None, is_minimum=False)
    else:
        res = -linprog(-c, A, b, bounds=(None, None)).fun
    if res <= d:
        return True
    else:
        print('-')
        print(res)
        print(d)
        return False


def remove_redundant_constraints_first_step(A, b, is_gurobi, stay_index):
    assert len(A) == len(b)
    for i in range(len(A)):
        if i in stay_index:
            continue
        if is_redundant(A[stay_index], b[stay_index], A[i], b[i], is_gurobi):
            pass
        else:
            stay_index.append(i)
        print(len(stay_index))
    return A[stay_index], b[stay_index]


def remove_redundant_constraints_last_step(A: np.array, b: np.array, is_gurobi, stay_index):
    assert len(A) == len(b)
    i = 0
    while i < len(A):
        if i in stay_index:
            continue
        if is_redundant(np.delete(A, i, 0), np.delete(b, i, 0), A[i], b[i], is_gurobi):
            A, b = np.delete(A, i, 0), np.delete(b, i, 0)
        else:
            i += 1
        print(len(A))
    return A, b


def redundant_constraints_remover(A, b, is_gurobi=True, stay_index=None):
    """
    :param A, b: The h-rep of a polytope;
    """
    A, b = remove_redundant_constraints_first_step(A, b, is_gurobi, stay_index)
    return remove_redundant_constraints_last_step(A, b, is_gurobi, stay_index)


def is_in_polytope(inputs: np.array, A: np.array, b: np.array):
    res = []
    for input in inputs:
        res.append((np.matmul(A, input) <= b).all())
    return res


def find_all_linear_regions(A: np.array, b: np.array, layers):
    """
    :param layers: the weights and bias of a neural network.
    :return: all the linear regions in area {x|Ax <= b}
    The linear regions have the form [curr_A, curr_b], which is defined as {x|curr_A x <= curr_b}.
    """
    todo = [[A, b, [] ]]
    all_linear_regions = []
    while todo:
        A, b, pattern = todo.pop()
        curr_layer = len(pattern)
        if curr_layer == len(layers) - 1:
            all_linear_regions.append([A, b, pattern])
            continue
        # C = np.eye(len(layers[0].weight[0]))
        # d = np.zeros(len(layers[0].weight[0]))
        # for l in range(curr_layer):
        #     weight, bias = np.matmul(np.diag(pattern[l]), layers[l].weight.data.numpy()), \
        #                    np.matmul(np.diag(pattern[l]), layers[l].bias.data.numpy())
        #     C = np.matmul(weight, C)
        #     d = np.matmul(weight, d) + bias
        # C = np.matmul(layers[curr_layer].weight.data.numpy(), C)
        # d = np.matmul(layers[curr_layer].weight.data.numpy(), d) + layers[curr_layer].bias.data.numpy()
        C, d = linearize_NN_from_pattern(pattern, layers[:curr_layer+1])

        curr_list = [[A, b, [] ]]
        while curr_list:
            curr_A, curr_b, curr_pattern = curr_list.pop()
            curr_neuron = len(curr_pattern)
            if curr_neuron == len(layers[curr_layer].bias):
                todo.append([curr_A, curr_b, pattern + [curr_pattern]])
                continue
            lower_bound, upper_bound = solve_LP(C[curr_neuron], curr_A, curr_b, bounds=None) + d[curr_neuron], \
                                       solve_LP(C[curr_neuron], curr_A, curr_b, bounds=None, is_minimum=False) + d[curr_neuron]
            if (upper_bound > 0) and (lower_bound < 0):
                curr_list.append([np.append(curr_A, [C[curr_neuron]], axis=0), np.append(curr_b, -d[curr_neuron]), curr_pattern + [0]])
                curr_list.append([np.append(curr_A, [-C[curr_neuron]], axis=0), np.append(curr_b, d[curr_neuron]), curr_pattern + [1]])
            elif upper_bound < 0:
                curr_list.append([curr_A, curr_b, curr_pattern + [0]])
            else:
                curr_list.append([curr_A, curr_b, curr_pattern + [1]])
    return all_linear_regions


def linearize_NN_from_pattern(pattern, layers):
    assert len(pattern) == len(layers) - 1
    C = np.eye(len(layers[0].weight[0]))
    d = np.zeros(len(layers[0].weight[0]))
    for l in range(len(layers) - 1):
        weight, bias = np.matmul(np.diag(pattern[l]), layers[l].weight.data.numpy()), \
                       np.matmul(np.diag(pattern[l]), layers[l].bias.data.numpy())
        C = np.matmul(weight, C)
        d = np.matmul(weight, d) + bias
    C = np.matmul(layers[len(layers) - 1].weight.data.numpy(), C)
    d = np.matmul(layers[len(layers) - 1].weight.data.numpy(), d) + layers[len(layers) - 1].bias.data.numpy()
    return C, d


def verify(A, b, pattern, layers, right_label=4):
    res = True
    P, ql, qu = [item[0] for item in specification_matrix_from_labels([right_label], 5)]
    C, d = linearize_NN_from_pattern(pattern, layers)
    for i in range(len(P)):
        # res = res and (solve_LP(np.matmul(P[i], C), A, b, None) + np.matmul(P[i], d) >= ql[i])
        res = res and (solve_LP(np.matmul(P[i], C), A, b, None, is_minimum=False) + np.matmul(P[i], d) <= qu[i])
    return res


if __name__ == '__main__':
    # class Model(nn.Module):
    #     def __init__(self):
    #         super(Model, self).__init__()
    #         self.l1 = nn.Linear(2, 4)
    #         self.l2 = nn.Linear(4, 4)
    #         self.l3 = nn.Linear(4, 1)
    #
    #     def forward(self, x):
    #         self.z1 = self.l1(x)
    #         self.z2 = self.l2(nn.ReLU()(self.z1))
    #         return self.l3(nn.ReLU()(self.z2))
    # model = Model()
    # x = torch.tensor([1.0, 2.0], requires_grad=True)
    # print(model(x), model.z1, model.z2)
    # all_neurons = torch.cat([model(x), model.z1, model.z2])
    # print(linear_region_from_input(x, all_neurons))
    # np.random.seed(24)
    # x = np.ones(3) * 0.3
    # A, b = np.random.rand(100, 3), np.random.rand(100)
    # for i in range(len(A)):
    #     if np.matmul(A[i], x) - b[i] > 0:
    #         A[i] = -A[i]
    #         b[i] = -b[i]
    # print(is_in_polytope([x], A, b))
    # start = time.time()
    # A1, b1 = remove_redundant_constraints_first_step(A, b)
    # print('First round: ', len(A1))
    # A2, b2 = remove_redundant_constraints_last_step(A1, b1)
    # print('Final round: ', len(A2))
    # print('Time cost: ', time.time() - start)
    # print(is_in_polytope([x], A2, b2))
    # A = np.concatenate([np.eye(3), -np.eye(3)], axis=0)
    # b = np.array([5000.0, 5000.0, -1/2*math.pi, -0.0, -0.0, math.pi])
    # model = HCAS_Model('TrainedNetworks/HCAS_rect_v6_pra1_tau05_25HU_3000.nnet')
    # all_linear_regions = find_all_linear_regions(A, b, model.layers)
    # print(len(all_linear_regions))
    # np.save('all_linear_regions.npy', all_linear_regions, allow_pickle=True)
    all_linear_regions = np.load('all_linear_regions.npy', allow_pickle=True)

