import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from ADMM_pMRF import Penalty, close_form, cp_solver_1, cp_solver_2
import cvxpy as cp
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor

# 模拟数据生成过程DGP
def DGP_1(n:int, p:int) -> list[np.ndarray, np.ndarray]:
    X = np.random.rand(n, p)
    X1 = X[:, 0].reshape(-1, 1)
    error = np.random.randn(n, 1)
    y = 10 * np.sin(math.pi * X1) + error
    return [X, y]

def DGP_2(n:int, p:int) -> list[np.ndarray, np.ndarray]:
    X = np.random.rand(n, p)
    X1 = X[:, 0].reshape(-1, 1)
    X2 = X[:, 1].reshape(-1, 1)
    X3 = X[:, 2].reshape(-1, 1)
    X4 = X[:, 3].reshape(-1, 1)
    X5 = X[:, 4].reshape(-1, 1)
    epsilon = np.random.randn(n, 1)
    y = 10 * np.sin(math.pi * X1 * X2) + 20 * (X3 - 0.05)**2 + 10 * X4 + 5 * X5 + epsilon
    return [X, y]

def DGP_3(n:int, p:int) -> list[np.ndarray, np.ndarray]:
    X = np.random.rand(n, p)
    X1 = X[:, 0].reshape(-1, 1)
    X2 = X[:, 1].reshape(-1, 1)
    X3 = X[:, 2].reshape(-1, 1)
    X4 = X[:, 3].reshape(-1, 1)
    X5 = X[:, 4].reshape(-1, 1)
    epsilon = np.random.randn(n, 1)
    y = np.zeros((n, 1))
    for i in range(n):
        ri = decision_tree(X1[i], X2[i], X3[i], X4[i], X5[i])
        y[i] = ri + epsilon[i]
    return [X, y]

def decision_tree(X1, X2, X3, X4, X5):
    if X4 < 0.383:
        if X2 < 0.2342:
            return 8.177
        else:
            if X1 < 0.2463:
                return 8.837
            else:
                return 13.15
    else:
        if X1 < 0.47:
            if X5 < 0.2452:
                return 10.99
            else:
                if X3 >= 0.2234:
                    return 18.03
                else:
                    return 13.87
        else:
            if X2 < 0.2701:
                return 15.02
            else:
                if X5 < 0.5985:
                    return 18.61
                else:
                    return 21.74

# 标准的随机森林拟合
def RF_train(X:np.ndarray, y:np.ndarray, M:int, max_depth:int, max_features:int) -> list[list[RandomForestRegressor], np.ndarray, np.ndarray, list[float]]:
    rf = RandomForestRegressor(n_estimators=M, max_depth=max_depth, max_features=max_features)
    rf.fit(X, y)
    trees = rf.estimators_
    trees_pred = np.zeros((X.shape[0], M))
    sigma2 = []
    for i, tree in enumerate(trees):
        trees_pred[:, i] = tree.predict(X)
        sigma2.append(np.var(tree.predict(X).reshape(-1, 1) - y))
    km = [tree.tree_.node_count for tree in trees]
    km = np.array(km).reshape(-1, 1)
    return [trees, trees_pred, km, sigma2]

def RF_predict(X:np.ndarray, trees:list[RandomForestRegressor]) -> np.ndarray:
    return np.mean([tree.predict(X) for tree in trees], axis=0).reshape(-1, 1)

def RF_predict_weighted(X:np.ndarray, trees:list[RandomForestRegressor], w:np.ndarray) -> np.ndarray:
    pres = np.array([tree.predict(X) for tree in trees])
    pres = pres.T
    return pres.dot(w).reshape(-1, 1)

# Boosted Trees模型
def boosted_trees(X:np.ndarray, y:np.ndarray, n_estimators:int, learning_rate:float) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
    model.fit(X, y)
    return model

# 集成模型（Random Forest + Boosted Trees）
def ensemble_model(X:np.ndarray, y:np.ndarray, rf_n_estimators:int, rf_max_features:int, gb_n_estimators:int, gb_learning_rate:float) -> VotingRegressor:
    rf_model = RandomForestRegressor(n_estimators=rf_n_estimators, max_features=rf_max_features)
    gb_model = boosted_trees(X, y, gb_n_estimators, gb_learning_rate)
    ensemble = VotingRegressor([('rf', rf_model), ('gb', gb_model)])
    ensemble.fit(X, y)
    return ensemble

# One Layer MLP模型
def one_layer_mlp(X:np.ndarray, y:np.ndarray, hidden_layer_sizes:tuple, max_iter:int) -> MLPRegressor:
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
    model.fit(X, y)
    return model

# 求解MRF
def cp_solver_MRF(y:np.ndarray, y_preds:np.ndarray, km:np.ndarray, sigma2_mean:float) -> np.ndarray:
    n, M = np.shape(y_preds)
    z = cp.Variable(M)
    y = y.ravel()
    km = km.ravel()
    y_hat = y_preds @ z
    cost = cp.sum_squares(y - y_hat) + 2 * sigma2_mean * cp.sum(km @ z)
    myprob = cp.Problem(cp.Minimize(cost), [cp.sum(z) == 1, z >= 0])
    myprob.solve()
    ans = z.value
    ans[abs(ans) < 1e-6] = 0
    ans = ans / sum(ans)
    return ans

# 计算目标函数的最优值
def grid_search(y:np.ndarray, y_preds:np.ndarray, km:np.ndarray, sigma2_mean:float, w_init:np.ndarray, lamb:float, method:str, tol=1e-4, K=1e3):
    alpha = 3.7
    theta = 0.1
    beta_list = [0.1, 1, 10, 100]
    cost_list = []
    w_list = []
    for beta in beta_list:
        w, z1, z2, x1, x2, count = ADMM_algorithm_pMRF(y, y_preds, km, sigma2_mean, w_init, lamb, beta, method, tol=tol, K=K)
        cost = np.sum((y - y_preds @ w)**2) + 2 * sigma2_mean * np.sum(km * w) + 2 * M * Penalty(alpha, theta, lamb, w, method)
        cost_list.append(cost)
        w_list.append(w)
    beta_opt = beta_list[np.argmin(cost_list)]
    w_opt = w_list[np.argmin(cost_list)]
    return beta_opt, w_opt, cost_list, w_list