from sklearn.model_selection import train_test_split
from pMRF_demo1 import *
import json
import numpy as np
from comparison3 import *
import os

# 创建存储结果的文件夹
if not os.path.exists("simulation_results7"):
    os.mkdir("simulation_results7")
    
# 参数设置 ；
# fixed
p = 100 
M = 1000
N = 50
method = 'TLP'

# vary
max_features = p + 0
# n_list = [100,200,300,400,500,750,1000]
n_list = [200,300,500]
# n_list = [400]
# max_depth_list = [10,20,30,50,100,None]
scenario_list = ['DGP1','DGP2','DGP3']
scenario_list = ['DGP3']
# max_features_list = [90,100]
max_features_list = [100]
max_depth = None 

# scenario = 'DGP_1'

# 新增模型参数配置
# TODO max_depth 和 base models 是否要调参数？
BOOSTED_TREES_PARAMS = {'n_estimators': 1000, 'max_depth': 300}
STACKING_PARAMS = {'base_models': None, 'final_estimator': LinearRegression()}
MLP_PARAMS = {'hidden_size': 50, 'max_iter': 1000}


for scenario in scenario_list:
    for n in n_list:
        for max_features in max_features_list:
            print("--------------------------------")
            print(f"n={n}, max_depth={max_depth}, max_features={max_features}")


            rmse_rf_list = []
            rmse_mrf_list = []
            rmse_pmrf_list = []
            rmse_bt_list = []    # Boosted Trees
            rmse_stk_list = []   # Stacking
            rmse_mlp_list = []   # MLP

            pmrf_best_lambda_list = []
            pmrf_count_list= []
            mrf_count_list = []



            for i in range(N):
                if scenario == 'DGP1':
                    X, y = DGP_1(n=n,p=p)
                elif scenario == 'DGP2':
                    X, y = DGP_2(n=n,p=p)
                elif scenario == 'DGP3':
                    X, y = DGP_3(n=n,p=p)

                # 分割训练集和测试集 
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
                X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

                # ===== 原有模型 =====
                # RF
                print("RF")
                [trees, trees_pred, km, sigma2, tree_depth, tree_leaf_num] = RF_train(X=X_train, y=y_train, M=1000, max_depth=None, max_features=max_features)  
                # 预测
                y_pred_rf = RF_predict(X=X_test, trees=trees)
                # 计算RMSE
                rmse_rf = np.sqrt(np.mean((y_test - y_pred_rf) ** 2))
                rmse_rf_list.append(rmse_rf)
                print(rmse_rf)
                
                # MRF
                print("MRF")
                sigma2_mean = np.mean(sigma2)
                w_mrf = cp_solver_MRF(y=y_train, y_preds=trees_pred, km=km, sigma2_mean=sigma2_mean)
                y_pred_mrf = RF_predict_weighted(X=X_test, trees=trees, w=w_mrf)
                rmse_mrf = np.sqrt(np.mean((y_test - y_pred_mrf) ** 2))
                rmse_mrf_list.append(rmse_mrf)
                print(rmse_mrf)
                mrf_count_list.append(float(np.sum(w_mrf>0)))


                # ===== 新增模型 =====
                # Boosted Trees
                print("Boosted Trees")
                bt_model = BoostedTrees_train(X_train, y_train, 
                                            **BOOSTED_TREES_PARAMS)
                y_pred_bt = BoostedTrees_predict(bt_model, X_test)
                rmse_bt = np.sqrt(np.mean((y_test - y_pred_bt) ** 2))
                rmse_bt_list.append(rmse_bt)
                print(rmse_bt)

                # Stacking
                print("Stacking")
                stk_model = Stacking_train(X_train, y_train, 
                                        **STACKING_PARAMS)
                print(stk_model.final_estimator_.coef_)
                y_pred_stk = Stacking_predict(stk_model, X_test)
                rmse_stk = np.sqrt(np.mean((y_test - y_pred_stk) ** 2))
                rmse_stk_list.append(rmse_stk)
                print(rmse_stk)

                # MLP
                print("MLP")
                mlp_model = MLP_train(X_train, y_train, 
                                    **MLP_PARAMS)
                y_pred_mlp = MLP_predict(mlp_model, X_test)
                rmse_mlp = np.sqrt(np.mean((y_test - y_pred_mlp) ** 2))
                rmse_mlp_list.append(rmse_mlp)
                print(rmse_mlp)

                # ===== Our Model =====
                # pMRF
                '''print("pMRF")
                w_init = w_mrf + 0
                best_lambda,best_w,w_opt_list,rmse_list,rmse_list_test = cv_search(X_val, X_test,y_train, y_val,y_test,trees,trees_pred,km,sigma2_mean,w_init,method)
                y_pred_pmrf = RF_predict_weighted(X=X_test, trees=trees, w=best_w)
                rmse_pmrf = np.sqrt(np.mean((y_test - y_pred_pmrf) ** 2))
                rmse_pmrf_list.append(rmse_pmrf)
                print(rmse_pmrf)
                pmrf_best_lambda_list.append(best_lambda)
                pmrf_count_list.append(float(np.sum(best_w>0)))'''

            # 结果存储
            results = {
                "rmse_rf_list": rmse_rf_list,
                "rmse_mrf_list": rmse_mrf_list,
                #"rmse_pmrf_list": rmse_pmrf_list,
                "rmse_bt_list": rmse_bt_list,
                "rmse_stk_list": rmse_stk_list,
                "rmse_mlp_list": rmse_mlp_list,
                "mrf_count_list": mrf_count_list,
                #"pmrf_count_list": pmrf_count_list,
                #"pmrf_best_lambda_list": pmrf_best_lambda_list,
                "params": {
                    "n": n,
                    "max_features": max_features,
                    "scenario": scenario,
                    "penalty_mehod":method,
                    "p": p,
                    "M": M,
                    "N": N,
                    "BOOSTED_TREES_PARAMS": str(BOOSTED_TREES_PARAMS),
                    "STACKING_PARAMS": str(STACKING_PARAMS),  # 避免JSON序列化问题
                    "MLP_PARAMS": str(MLP_PARAMS)
                }
            }
            # 保存为 JSON 文件
            file_name = f"simulation_results7/rmse_{scenario}_n{n}_max_features_{max_features}.json"
            with open(file_name, "w") as f:
                json.dump(results, f)
