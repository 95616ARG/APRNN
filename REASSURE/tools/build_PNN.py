import torch, time, torch.nn as nn, numpy as np
from REASSURE.tools.models import MLPNet
from REASSURE.tools.gurobi_solver import solve_LP
# from scipy.optimize import linprog
from REASSURE.tools.linear_region import linear_region_from_input, redundant_constraints_remover, is_in_polytope, linearize_NN_from_pattern
from torch import multiprocessing


class SupportNN(nn.Module):
    def __init__(self, A: np.ndarray, b: np.ndarray, n):
        """
        SupportNN is a neural network that almost only active on polytope {x|Ax<=b}.
        :param A, b: H-rep of a polytope.
        :param n: To control the side effect. Choose a large number until the 100% repair rate.
        """
        # Ax <= b
        super(SupportNN, self).__init__()
        assert len(A) == len(b)
        A = torch.tensor(A)
        b = torch.tensor(b)
        assert len(A.size()) == 2
        layer = nn.Linear(*A.size())
        layer.weight = torch.nn.Parameter(-A)
        layer.bias = torch.nn.Parameter(b)
        self.layer, self.l, self.n = layer, len(A), n

    def forward(self, x):
        s = - self.l + 1
        y = self.layer(x)
        s += torch.sum(nn.ReLU()(self.n * y + 1), -1, keepdim=True) - torch.sum(nn.ReLU()(self.n * y), -1, keepdim=True)
        return nn.ReLU()(s)


class PatchNN(nn.Module):
    def __init__(self, g, c, d):
        super(PatchNN, self).__init__()
        self.K = torch.max(torch.max(torch.max(c, torch.tensor(0.0)).sum(1)+d),
                           torch.max(torch.max(-c, torch.tensor(0.0)).sum(1)-d)).squeeze()
        self.layer = nn.Linear(*c.size())
        self.layer.weight = torch.nn.Parameter(c)
        self.layer.bias = torch.nn.Parameter(d)
        self.g = g

    def forward(self, x):
        return nn.ReLU()(self.layer(x) + self.K*self.g(x)-self.K) \
               - nn.ReLU()(-self.layer(x) + self.K*self.g(x)-self.K)


class MultiPNN(nn.Module):
    def __init__(self, g_list, cd_list, bounds):
        super(MultiPNN, self).__init__()
        import time
        from tqdm.auto import tqdm
        self.K_list, self.layer_list = [], []
        temp_c, temp_d = 0, 0
        bound_A = np.concatenate([-np.eye(len(bounds[0])), np.eye(len(bounds[1]))])
        bound_b = np.concatenate([-bounds[0], bounds[1]])
        for c, d in tqdm(cd_list, desc="MulitPNN"):
            c, d = c - temp_c, d - temp_d
            temp_c, temp_d = c + temp_c, d + temp_d
            K = torch.tensor(0.0)
            for i in range(len(c)):
                # out_lowerbound = linprog(c[i], bound_A, bound_b, bounds=[None, None]).fun + d[i]
                out_lowerbound = solve_LP(c[i].detach().numpy(), bound_A, bound_b, bounds=None)+d[i]
                # out_upperbound = -linprog(-c[i], bound_A, bound_b, bounds=[None, None]).fun + d[i]
                out_upperbound = solve_LP(c[i].detach().numpy(), bound_A, bound_b, is_minimum=False, bounds=None)+d[i]
                abs_bound = max(abs(out_upperbound), abs(out_lowerbound))
                K = torch.max(K, torch.tensor(abs_bound, dtype=torch.float))
            self.K_list.append(nn.Parameter(K))
            layer = nn.Linear(*c.size())
            layer.weight = torch.nn.Parameter(c)
            layer.bias = torch.nn.Parameter(d)
            self.layer_list.append(layer)
        # import pdb; pdb.set_trace()
        self.layer_list = nn.ModuleList(self.layer_list)
        self.K_list = nn.ParameterList(self.K_list)
        self.g_list = nn.ModuleList(g_list)

    def forward(self, x):
        # x = x.view(-1, 28 * 28)
        res = 0
        for i in range(len(self.g_list)):
            if i < len(self.g_list) - 1:
                y = [self.g_list[k](x) for k in range(i, len(self.g_list))]
                g_max = torch.max(torch.cat(y, dim=-1), dim=-1).values.unsqueeze(1)
            else:
                g_max = self.g_list[i](x)
            res += nn.ReLU()(self.layer_list[i](x) + self.K_list[i]*g_max-self.K_list[i]) \
               - nn.ReLU()(-self.layer_list[i](x) + self.K_list[i]*g_max-self.K_list[i])
        return res


def OrthogonalComplement(P):
    n, m = len(P), len(P[0])
    if n >= m: return P
    orthogonal_space = []
    P = np.concatenate([P, np.zeros([m-n, m])], axis=0)
    w, v = np.linalg.eig(P)
    epsilon = 0.1**5
    for i in range(m):
        if np.abs(w[i]) < epsilon:
            orthogonal_space.append(v[i]/np.linalg.norm(v[i], 2))
    P[n:] = orthogonal_space
    return P


def LinearPatchNN(A: np.ndarray, b: np.ndarray, P: np.ndarray, ql: np.ndarray, qu: np.ndarray, f1, f2, my_dtype=torch.float):
    """
    :param A, b: H-rep of a polytope(linear region);
    :param P, ql, qu: Specification is ql[i] <= P[i]model(x)[i] <= qu;
    :param f1, f2: In {x|Ax <= b}, model(x) = f1*x + f2;
    :param my_dtype:
    :return:
    """
    l_P = len(P)
    assert np.linalg.matrix_rank(P) == l_P
    P = OrthogonalComplement(P)
    P_inv = np.linalg.inv(P)
    alpha = np.eye(len(P[0]))
    beta = np.zeros(len(P[0]))
    for i in range(l_P):
        # # check alpha:
        # print(linprog(np.matmul(P[i], f1), A, b, bounds=(0, 1)).fun)
        # print(-linprog(-np.matmul(P[i], f1), A, b, bounds=(0, 1)).fun)

        # tem1 = linprog(np.matmul(P[i], f1), A, b, bounds=(None, None)).fun
        tem1 = solve_LP(np.matmul(P[i], f1), A, b, bounds=None)
        # tem2 = -linprog(-np.matmul(P[i], f1), A, b, bounds=(None, None)).fun
        tem2 = solve_LP(np.matmul(P[i], f1), A, b, bounds=None, is_minimum=False)

        alpha[i, i], beta[i] = OneDimLinearTransform(tem1, tem2, np.matmul(P[i], f2), ql[i], qu[i])
        # # check beta:
        # print(alpha[i, i] * (linprog(np.matmul(P[i], f1), A, b, bounds=(0, 1)).fun + np.matmul(P[i], f2)) + beta[i])
        # print(alpha[i, i] * (-linprog(-np.matmul(P[i], f1), A, b, bounds=(0, 1)).fun + np.matmul(P[i], f2)) + beta[i])
    trans_matrix = np.matmul(P_inv, np.matmul(alpha, P)) - np.eye(len(P[0]))
    c = np.matmul(trans_matrix, f1)
    d = np.matmul(trans_matrix, f2) + np.matmul(P_inv, beta)
    # print(linprog(np.matmul(P[0], f1+c), A, b, bounds=(0, 1)).fun)
    # print(-linprog(-np.matmul(P[0], f1+c), A, b, bounds=(0, 1)).fun)
    # print(np.matmul(P[0], f2+d))
    # print(np.matmul(P[0], f2+d) - np.matmul(alpha[0], np.matmul(P, f2)) - beta[0])
    return torch.tensor(c, dtype=my_dtype), torch.tensor(d, dtype=my_dtype)


def OneDimLinearTransform(lb, ub, b, ql, qu):
    # [lb, ub]: objective interval, b:
    # [ql, qu]: require interval
    if ub - lb <= qu - ql:
        alpha = 1.0
    else:
        alpha = (qu - ql)/(ub - lb)
        lb, ub = alpha*lb, alpha*ub
    if lb + b >= ql and ub + b <= qu:
        beta = 0.0
    elif lb + b < ql:
        beta = ql - lb - alpha*b
    else:
        beta = qu - ub - alpha*b
    return alpha, beta


def LinearizedNN(input, model):
    """
    :param input: x;
    :param model: neural network;
    :return: f1, f2 where model(x) = f1*x+f2 on the linear region.
    """
    f1, f2 = [], []
    for output in model(input).squeeze():
        model(input)
        output.backward(retain_graph=True)
        f1.append(input.grad.clone().detach())
        f2.append(output - torch.matmul(input.grad, input))
    # print(torch.matmul(torch.stack(f1), input) + torch.stack(f2) - model(input))
    return torch.stack(f1).detach().numpy(), torch.stack(f2).detach().numpy()


class MultiPointsPNN:
    def __init__(self, model, n, bounds, test_model=False):
        """
        Multiple points PNN repair for a neural network, return a repaired neural network.
        :param inputs: tensor of inputs that need to be repaired;
        :param model: Pytorch neural network and model should have a function all_hidden_neurons to list all the hidden neurons value for a input;
        :param P, ql, qu: Specification is ql[i] <= P[i]model(x)[i] <= qu;
        :param n: To control the side effect. Choose a large number until the 100% repair rate;
        :param bounds: bounds for neural network inputs.
        bounds = [lowerbound_list, upperbound_list], lowerbound is a list of lowerbound for every dimension;
         For MNIST, the input should be float between [0, 1]^D or int between [0, 255]^D;
        :param remove_redundant_constraint: take True if you want to remove redundant constraints for a H-rep (linear region),
         before build support neural networks;
        :param test_model: take True if you want to print some results and debug.
        """
        self.model = model
        self.n = n
        self.bounds = bounds
        self.test_model = test_model
        self.g_list, self.cd_list = [], []
        self.total_constraint_num = []
        self.buggy_points = None
        self.linear_regions = None
        self.P = None
        self.ql = None
        self.qu = None
        self.remove_redundant_constraint = None

    def point_wise_repair(self, buggy_points, P, ql, qu, remove_redundant_constraint):
        assert self.linear_regions is None
        self.buggy_points = buggy_points
        self.P, self.ql, self.qu = P, ql, qu
        self.remove_redundant_constraint = remove_redundant_constraint

    def PatchForOnePoint(self, nn_input: torch.Tensor, P_i, ql_i, qu_i, is_gurobi):
        nn_input = nn_input.view(-1)
        nn_input.requires_grad = True
        A, b = linear_region_from_input(nn_input, self.model.all_hidden_neurons(nn_input).view(-1), self.bounds)
        x_dim = len(A[0])
        if self.remove_redundant_constraint:
            A, b = redundant_constraints_remover(A, b, is_gurobi, stay_index=list(range(2*x_dim)))  # A, b are np.ndarray
        simplified_A, simplified_b = A[2*x_dim:], b[2*x_dim:]
        support_nn = SupportNN(simplified_A, simplified_b, self.n)
        f1, f2 = LinearizedNN(nn_input, self.model)  # f1, f2 are np.ndarray
        cd = LinearPatchNN(A, b, P_i, ql_i, qu_i, f1, f2)
        return support_nn, cd, len(simplified_A)  # len(simplified_A) is for computation of parameters overhead

    def area_repair(self, linear_regions, P, ql, qu):
        assert self.buggy_points is None
        self.linear_regions = linear_regions
        self.P, self.ql, self.qu = P, ql, qu

    def PatchForOneLinearRegion(self, A, b, pattern, P_i, ql_i, qu_i):
        support_nn = SupportNN(A.astype(np.float32), b.astype(np.float32), self.n)
        f1, f2 = linearize_NN_from_pattern(pattern, self.model.layers)
        cd = LinearPatchNN(A.astype(np.float32), b.astype(np.float32), P_i, ql_i, qu_i, f1, f2)
        return support_nn, cd, len(A)

    def compute(self, core_num=multiprocessing.cpu_count()-1, is_gurobi=True):
        assert is_gurobi == True
        assert (self.buggy_points is not None) or (self.linear_regions is not None)
        print('Working on {} cores.'.format(core_num))
        multiprocessing.set_start_method('spawn', force=True)
        self.model.share_memory()
        pool = multiprocessing.Pool(core_num)
        if self.buggy_points is not None:
            from tqdm.auto import tqdm
            arg_list = [[self.buggy_points[i], self.P[i], self.ql[i], self.qu[i], is_gurobi] for i in
                        range(len(self.buggy_points))]
            res = tqdm(pool.starmap(self.PatchForOnePoint, arg_list))
        else:
            assert False
            arg_list = [[self.linear_regions[i][0], self.linear_regions[i][1], self.linear_regions[i][2], self.P[i],
                         self.ql[i], self.qu[i]] for i in range(len(self.linear_regions))]
            res = pool.starmap(self.PatchForOneLinearRegion, arg_list)
        pool.close()
        g_list, cd_list = [res_g[0] for res_g in res], [res_g[1] for res_g in res]
        self.total_constraint_num = [res_g[2] for res_g in res]
        h = MultiPNN(g_list, cd_list, self.bounds)
        print('avg_constraint_num:', sum(self.total_constraint_num) / len(self.total_constraint_num))
        return NNSum(self.model, h)


class NNSum(torch.nn.Module):
    def __init__(self, target_nn: torch.nn.Module, pnn: torch.nn.Module):
        super(NNSum, self).__init__()
        self.target_nn = target_nn
        self.pnn = pnn

    def forward(self, x):
        if len(x.size()) >= 3:
            pnn_out = self.pnn(x.view(len(x), -1))
        else:
            pnn_out = self.pnn(x)
        return self.target_nn(x)+pnn_out


###

# def PointWisePNN(input, model, P: np.array, ql: np.array, qu: np.array, n, bounds, remove_redundant_constraint=True):
#     # input: require_grads = True
#     # model must have att all_neurons;
#     A, b = linear_region_from_input(input, model.all_hidden_neurons(input), bounds)
#     print('We have {} constraints.'.format(len(A)))
#     if remove_redundant_constraint:
#         start = time.time()
#         A, b = redundant_constraints_remover(A, b)
#         print('Time cost for remove redundant constraints: ', time.time() - start)
#         print('We have {} constraints after simplification.'.format(len(A)))
#     start = time.time()
#     g = SupportNN(A, b, n)
#     # print(g(input))
#     f1, f2 = LinearizedNN(input, model)
#     c, d = LinearPatchNN(A, b, P, ql, qu, f1, f2)
#     h = PatchNN(g, c, d)
#     print('Time cost for PatchNN: ', time.time() - start)
#     return NNSum(model, h)


if __name__ == '__main__':
    # test SupportNN and PatchNN
    # A = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    # b = torch.tensor([-0.3, -0.3, 0.5, 0.5])
    # c = torch.tensor([0.3, 0.6])
    # d = torch.tensor([0.0])
    # g = SupportNN(A, b, 10)
    # h = PatchNN(g, c, d)
    # from tools.other_tools import Plot_3D
    # x, y = torch.linspace(-2, 2, 101), torch.linspace(-2, 2, 101)
    # grid_x, grid_y = torch.meshgrid(x, y)
    # xy = torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)], dim=-1)
    # z = h(xy)
    # Plot_3D(grid_x.detach().numpy(), grid_y.detach().numpy(), z.reshape(*grid_x.size()).detach().numpy())
    # print(h(torch.tensor([0.0, 2.1])))

    # # test LinearPatchNN
    # A, b = np.concatenate([np.eye(2), -np.eye(2)], axis=0), np.array([1.0, 3.0, 1, 3])
    # # -1 <= x_1 <= 1, -3 <= x_2 <= 3
    # P = np.array([[1.0, 0.0]])
    # ql, qu = np.array([0]), np.array([2])
    # # 0 <= x_1 <= 2
    # f1, f2 = np.eye(2), np.zeros(2)
    # c, d = LinearPatchNN(A, b, P, ql, qu, f1, f2)
    # print(f1, f2)
    # print(f1+c, f2+d)

    # # Test PointWisePNN:
    # class Model(nn.Module):
    #     def __init__(self):
    #         super(Model, self).__init__()
    #         self.l1 = nn.Linear(2, 4)
    #         self.l2 = nn.Linear(4, 4)
    #         self.l3 = nn.Linear(4, 2)
    #
    #     def forward(self, x):
    #         self.z1 = self.l1(x)
    #         self.z2 = self.l2(nn.ReLU()(self.z1))
    #         return self.l3(nn.ReLU()(self.z2))
    #
    #     def all_hidden_neurons(self, x):
    #         self(x)
    #         return torch.cat([self.z1, self.z2])
    # model = Model()
    # input = torch.tensor([0.3, 0.5], requires_grad=True)
    # print(model(input))
    # P = np.eye(2)
    # ql, qu = np.array([0.0, 0.0]), np.array([0.1, 0.1])
    # n = 100
    # h = PointWisePNN(input, model, P, ql, qu, n)
    # print(model(input) + h(input))

    # # Test OrthogonalComplement
    # P = np.random.rand(3, 10)
    # print(OrthogonalComplement(P))

    # # Test MultiPointsPNN:
    # class Model(nn.Module):
    #     def __init__(self):
    #         super(Model, self).__init__()
    #         self.l1 = nn.Linear(2, 64)
    #         self.l2 = nn.Linear(64, 64)
    #         self.l3 = nn.Linear(64, 2)
    #
    #     def forward(self, x):
    #         self.z1 = self.l1(x)
    #         self.z2 = self.l2(nn.ReLU()(self.z1))
    #         return self.l3(nn.ReLU()(self.z2))
    #
    #     def all_hidden_neurons(self, x):
    #         self(x)
    #         return torch.cat([self.z1, self.z2])
    # model = Model()
    n = 4
    # inputs = torch.rand([n, 2])
    # print(model(inputs))
    # P = [np.eye(2)]*n
    # ql, qu = [np.array([0.0, 0.0])]*n, [np.array([0.1, 0.1])]*n
    # n = 100
    # h = MultiPointsPNN(inputs, model, P, ql, qu, n, bounds=[np.zeros(2), np.ones(2)])
    # print(model(inputs) + h(inputs))
    #
    # print(h(torch.rand([100, 2])))

