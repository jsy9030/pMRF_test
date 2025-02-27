#  Terry 2024-12-15
#  pMRF 那篇论文的所有代码

# 1. 模拟数据生成过程DGP
# 2. 标准RF 的求解 
# 3. MRF 即 带mallows准则的RF  用自带的求解器求解
# 4. pMRF 即 带惩罚的MRF 要用ADMM算法求解  单独写在ADMM_pMRF.py中


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time 
import math
from sklearn.model_selection import train_test_split
from ADMM_pMRF import Penalty, close_form, cp_solver_1, cp_solver_2

# 可能不太好安装 maybe无法直接pip
import cvxpy as cp

from sklearn.ensemble import RandomForestRegressor

def DGP_1(n:int, p:int) -> list[np.ndarray, np.ndarray]:
    """
    模拟数据生成过程DGP
    :Biau, G. (2012). Analysis of a random forests model. The Journal of Machine Learning Research, 13(1):1063-1095
    :X ~ U[0, 1]
    :y = 10 * sin(pi * X1) + epsilon
    """
    X = np.random.rand(n, p)
    X1 = X[:,0].reshape(-1,1)
    # error 的shape 要和 X1 一致
    error = np.random.randn(n, 1)
    y = 10 * np.sin(math.pi * X1 ) + error

    return [X, y]

def DGP_2(n:int, p:int) -> list[np.ndarray, np.ndarray]:
    """
    模拟数据生成过程DGP 2
    :y = 10 sin(pi * X1 * X2) + 20*(X3 - 0.05)**2 + 10*X4 + 5*X5 + epsilon
    """

    X = np.random.rand(n,p)
    # 把X 转成 (n,1) 的shape
    X1 = X[:,0].reshape(-1,1)
    X2 = X[:,1].reshape(-1,1)
    X3 = X[:,2].reshape(-1,1)
    X4 = X[:,3].reshape(-1,1)
    X5 = X[:,4].reshape(-1,1)
    epsilon = np.random.randn(n,1)

    y = 10 * np.sin(math.pi * X1 * X2) + 20*(X3 - 0.05)**2 + 10*X4 + 5*X5 + epsilon

    return [X, y]

def DGP_3(n:int, p:int) -> list[np.ndarray, np.ndarray]:
    """
    模拟数据生成过程DGP 3
    """
    X = np.random.rand(n,p)
    # 把X 转成 (n,1) 的shape
    X1 = X[:,0].reshape(-1,1)
    X2 = X[:,1].reshape(-1,1)
    X3 = X[:,2].reshape(-1,1)
    X4 = X[:,3].reshape(-1,1)
    X5 = X[:,4].reshape(-1,1)
    epsilon = np.random.randn(n,1)

    y = np.zeros((n,1))
    for i in range(n):
        ri = decision_tree(X1[i],X2[i],X3[i],X4[i],X5[i])
        y[i] = ri + epsilon[i]

    return [X,y]


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

 

def plot_simulation_data(X1:np.ndarray, y:np.ndarray):
    plt.scatter(X1, y)
    plt.show()

def plot_pred_result(X1:np.ndarray, y:np.ndarray, y_pred:np.ndarray):
    plt.scatter(X1, y)
    plt.scatter(X1, y_pred, color='red', alpha=0.5)
    plt.show()

# 标准的随机森林拟合
def RF_train(X:np.ndarray, y:np.ndarray, M:int, max_depth:int, max_features:int) -> list[list[RandomForestRegressor], np.ndarray, np.ndarray,list[float]]:
    """
    随机森林拟合
    """
    rf = RandomForestRegressor(n_estimators=M, max_depth=max_depth, max_features=max_features)
    rf.fit(X, y)
    # 返回每个树
    trees = rf.estimators_
    # 返回每个树的深度
    tree_depth = [tree.tree_.max_depth for tree in trees]
    # 有多少叶子结点
    tree_leaf_num = [tree.tree_.node_count for tree in trees]

    # 返回每个树的预测值
    trees_pred = np.zeros((X.shape[0], M))
    sigma2 = []
    for i, tree in enumerate(trees):
        trees_pred[:, i] = tree.predict(X)
        sigma2.append(np.var(tree.predict(X).reshape(-1, 1) - y))

    # 返回每个树的根结点和叶结点的个数 即km 并且转成和y一样的shape
    km = [tree.tree_.node_count for tree in trees]
    km = np.array(km).reshape(-1, 1)


    return [trees, trees_pred, km, sigma2, tree_depth, tree_leaf_num]

# RF的预测
def RF_predict(X:np.ndarray, trees:list[RandomForestRegressor]) -> np.ndarray:
    """
    RF的预测
    """
    return np.mean([tree.predict(X) for tree in trees], axis=0).reshape(-1,1)


# 带权重的RF的预测
def RF_predict_weighted(X:np.ndarray, trees:list[RandomForestRegressor], w:np.ndarray) -> np.ndarray:
    """
    带权重的RF的预测
    MRF和pMRF都可用
    """
    pres = np.array([tree.predict(X) for tree in trees])
    # pres shape : M*n
    pres = pres.T
    ans = pres.dot(w)
    return ans.reshape(-1,1)

# 求解MRF
def cp_solver_MRF(y:np.ndarray, y_preds:np.ndarray, km:np.ndarray, sigma2_mean:float) -> np.ndarray:
    """
    Mallows RF
    :目标函数为 (y- sum(wi*y_pred_i))**2 +  2 * sigma2_mean * sum(wi * km)
    :利用python自带的cvxpy求解器求解
    
    Shape:
    : y : (n, 1)
    : y_pred : (n, M)
    : km : (M, 1)
    : ans : (M, 1)
    """

    # print(y.shape, y_preds.shape, km.shape, sigma2_mean.shape)
    # print(type(y), type(y_preds), type(km), type(sigma2_mean))

    n,M = np.shape(y_preds)

    z = cp.Variable(M)
    # 由于z的shape 是 (M,)
    # 所以把 y 和 km 转一下
    y = y.ravel()
    km = km.ravel()
    # print(y.shape, km.shape)

    y_hat = y_preds@z
    cost = cp.sum_squares(y - y_hat) + 2 * sigma2_mean * cp.sum(km @ z)

    myprob = cp.Problem(cp.Minimize(cost), [cp.sum(z) == 1, z >= 0])
    myprob.solve()
    # print("MRF的解: ", prob.value)
    # print("z的解: ", z.value)

    # 特别小的先给压成0
    ans = z.value
    ans[abs(ans)<1e-10] = 0 
    ans = ans/sum(ans)
    return (ans)



def ADMM_algorithm_pMRF(y:np.ndarray,y_preds:np.ndarray,km:np.ndarray,sigma2_mean:float,w_init:np.ndarray,lamb:float,beta:float,method:str,tol = 1e-4,K = 1e3 ,beta_vary = True):
    """
    利用ADMM 求解 pMRF
    :y 真实值 shape (n,1)
    :y_pred 随机森林的预测结果  shape (n,M)
    :km 每个树中根结点和叶结点的个数  shape (M,1)
    :sigma2_mean 每个树的sigma2的均值
    :w_init 初始权重  shape (M,1)
    : lamb 惩罚系数
    : ADMM 算法中的参数
    : tol 迭代停止条件
    : K 迭代次数
    """

    if method in ['SCAD','MCP','TLP']:
        pass
    else:
        raise ValueError("method 只能是 SCAD, MCP, TLP")

    # 处理shape
    km = km.ravel()
    w_init = w_init.ravel()

    # init param
    n,p = np.shape(y_preds)
    w = w_init + 0
    x1 = w_init + 0
    x2 = w_init + 0
    z1_old = w_init + 0
    z2_old = w_init + 0
    

    error = 1
    count = 0
    while error > tol and count < K:
        count += 1

        # 式3.11的迭代方法
        for i in range(p):
            x1i = x1[i]
            z1i = z1_old[i]
            x2i = x2[i]
            z2i = z2_old[i]
            
            #只有 这里 closeform 的时候 不同penalty的不同 后面的cp solver 是一样的
            w[i] = close_form(lamb,x1i,z1i,x2i,z2i,beta,method) + 0
        
        # print(w)
        # 注意矩阵运算 shape要完全一致
        z1_new = cp_solver_1(y,y_preds,km,sigma2_mean,beta,w,x1)
        x1 = x1 + beta*(z1_new-w)

        z2_new = cp_solver_2(y,y_preds,km,sigma2_mean,beta,w,x2)
        x2 = x2 + beta*(z2_new-w)

        prime_residuals = np.sum(np.abs(z1_new-w)) +  np.sum(np.abs(z2_new-w))
        dual_residuals = np.sum(np.abs(z1_old-z1_new)) +  np.sum(np.abs(z2_old-z2_new))

        error = prime_residuals + dual_residuals

        z1_old = z1_new + 0
        z2_old = z2_new + 0

        # print(prime_residuals,dual_residuals,error,count)

        # 根据prime_residuals和dual_residuals更新beta
        """
        参考Boyd 2010 ADMM 中的 3.4.1 Varying Penalty Parameter
        """
        if beta_vary:
            if prime_residuals > 10 * dual_residuals:
                # print("beta 增大")
                beta = beta * 2
            elif prime_residuals < 0.1 * dual_residuals:
                # print("beta 减小")
                beta = beta * 0.5


    w[abs(w)<1e-6] = 0 
    w = w/sum(w)
    print(error,count)
    
    return(w,z1_new,z2_new,x1,x2,count)


def grid_search(y:np.ndarray,y_preds:np.ndarray,km:np.ndarray,sigma2_mean:float,w_init:np.ndarray,lamb:float,method:str,tol = 1e-4,K = 1e4):
    """
    给定不同的init-beta网格搜索
    : 如果 模拟中 ADMM_algorithm_pMRF 无法稳定的收敛的结果 就进行网格搜索 即给多个init-beta 进行求解
    : 并按照目标函数的最优值 进行对应的beta 。反之如果每次ADMM_algorithm_pMRF都可以求解, 就不用这个grid search 毕竟速度上更慢
    : beta_list可能需要拍脑袋确定,但由于有vary的机制, 其实也还好。
    """

    n,M = np.shape(y_preds)


    # penalty的 a or tau 用hard code 或者 外部传入都可以 比如：
    alpha = 3.7
    theta  =  0.05
    # hard code 给 或者 外部传入哦度可以
    beta_list = [100,300,1000,3000]

    count_list = []
    cost_list = []
    w_list = []
    for beta in beta_list:
        w,z1,z2,x1,x2,count = ADMM_algorithm_pMRF(y,y_preds,km,sigma2_mean,w_init,lamb,beta,method,tol = tol,K = K)

        # 计算目标函数的最优值
        cost = np.sum((y - y_preds*w)**2) + 2 * sigma2_mean * np.sum(km * w) + 2*M*Penalty(alpha,theta,lamb,w,method)
        # 记录目标函数的最优值
        cost_list.append(cost)
        w_list.append(w)
        count_list.append(count)
    # 找到目标函数的最优值对应的beta 和 w
    beta_opt = beta_list[np.argmin(cost_list)]
    w_opt = w_list[np.argmin(cost_list)]
    print(count_list)

    return beta_opt, w_opt,cost_list,w_list,count_list

def cv_search(X_val, X_test,y_train, y_val,y_test,trees,trees_pred,km,sigma2_mean,w_init,method):
    # 
    lambda_list = [0.0001,0.0005,0.001,0.002,0.003,0.005,0.01,0.02,0.03,0.05,0.1]
    rmse_list = []
    w_opt_list = []
    rmse_list_test = []
    for lambda_ in lambda_list:
        print('-'*10)
        print(lambda_)
        beta_opt, w_opt,cost_list,w_list,count_list = grid_search(y_train,trees_pred,km,sigma2_mean,w_init,lambda_,method,tol = 1e-3,K = 300)
        w_opt_list.append(w_opt)

        # 在val上的表现
        y_pred_val = RF_predict_weighted(X=X_val, trees=trees, w=w_opt)
        rmse = np.sqrt(np.mean((y_val - y_pred_val)**2))
        rmse_list.append(rmse)


        # 在test 上的表现 
        y_pred_test = RF_predict_weighted(X=X_test, trees=trees, w=w_opt)
        rmse_test = np.sqrt(np.mean((y_test - y_pred_test)**2))
        rmse_list_test.append(rmse_test)




    idx = np.argmin(np.array(rmse_list))
    best_lambda = lambda_list[idx]
    best_w = w_opt_list[idx]

    return(best_lambda,best_w,w_opt_list,rmse_list,rmse_list_test)

#  σ2 is an estimation of σ2 given the training sample and
#  km is the number of variables used in constructing the m-th tree predictor, indicating the degree of freedom of the m-th tree.
#  km 代表树中 根结点和叶结点 的个数

def calc_rmse(X,y,trees,w):
    y_pred = RF_predict_weighted(X=X, trees=trees, w=w)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    return  rmse


if __name__ == "__main__":
    # X, y = DGP_1(n=200,p=100)
    # X, y = DGP_2(n=200,p=100)
    X, y = DGP_3(n=400,p=100)

    # X和y的关系图
    # plot_simulation_data(X1, y)

    # 
    # split data 
    # 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

    # 随机森林拟合
    [trees, trees_pred, km, sigma2, tree_depth, tree_leaf_num] = RF_train(X=X_train, y=y_train, M=1000, max_depth=None, max_features=100)
    
    # 在val上的预测结果


    # # 画图 pred结果
    # y_pred = trees_pred[:,0]
    # y_preds = trees_pred
    # plot_pred_result(X1, y, y_pred)


    # 计算每个树中根结点和叶结点的个数 作为 mallows准则里的km
    km2 = [tree.tree_.node_count for tree in trees]
    print("每个树中根结点和叶结点的个数: ", km2)

    # print(km)
    print("每个树的sigma2: ", sigma2)
    sigma2_mean = np.mean(sigma2)
    print("sigma2的均值: ", sigma2_mean)

    # 确认一下y的shape 和 每个y_pred的shape
    print("y的shape: ", y.shape)
    print("每个y_pred的shape: ", trees_pred.shape)


    # 求解MRF
    z_value = cp_solver_MRF(y_train, trees_pred, km, sigma2_mean)
    print(f"shape of z_value :{z_value.shape}")
    print(f"number of active weight z_value :{np.sum(z_value>0)}")
    print(calc_rmse(X_test,y_test,trees,z_value))

    # 求解pMRF
    # 给lambda
    # 给beta
    # 给method

    lamb =  0.01
    beta = 1000
    method = 'TLP'

    w_init = z_value + 0
    w,z1,z2,x1,x2,count = ADMM_algorithm_pMRF(y_train,trees_pred,km,sigma2_mean,w_init,lamb,beta,method,tol = 1e-3,K = 300)
    # print(w)
    print(count)
    print(f"number of active weight :{np.sum(w>0)}")
    print(np.sum(w>0))

    t1 = time.time()
    # beta_opt, w_opt,cost_list,w_list,count_list = grid_search(y_train,trees_pred,km,sigma2_mean,w_init,lamb,method,tol = 1e-3,K = 300)
    # # # print(w_opt)
    # # print(count_list)
    # print(beta_opt)
    # print(f"number of active weight :{np.sum(w_opt>0)}")
    t2 = time.time()
    print(f"cost of time :{t2 - t1}")
    # print(calc_rmse(X_test,y_test,trees,w_opt))
    # 
    best_lambda,best_w,w_opt_list,rmse_list,rmse_list_test = cv_search(X_val, X_test,y_train, y_val,y_test,trees,trees_pred,km,sigma2_mean,w_init,method)
    t3 = time.time()
    print(t3 - t2)


    for w in w_opt_list:
        print(np.sum(w>0))


    print(rmse_list)
    print(rmse_list_test)
    print(calc_rmse(X_test,y_test,trees,z_value))
    M = 1000
    print(calc_rmse(X_test,y_test,trees,np.ones(M)/M))

    # now the problem is that 
    # active number of init result z_value is less than that 
    # the theory result should be init result equal to that when lambda -> 0 