from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# 1. Boosted Trees (Gradient Boosting)
def BoostedTrees_train(X: np.ndarray, y: np.ndarray, 
                      n_estimators: int = 1000, 
                      max_depth: int = 300,
                      learning_rate: float = 0.01) -> GradientBoostingRegressor:
    """梯度提升树训练"""
    model = GradientBoostingRegressor(n_estimators=n_estimators, 
                                     max_depth=max_depth,
                                     learning_rate = learning_rate,
                                     random_state=0)
    model.fit(X, y.ravel())
    return model

def BoostedTrees_predict(model: GradientBoostingRegressor, 
                        X: np.ndarray) -> np.ndarray:
    """梯度提升树预测"""
    return model.predict(X).reshape(-1, 1)

# 2. Ensemble Model (Stacking)
def Stacking_train(X: np.ndarray, y: np.ndarray,
                  base_models: list = None,
                  final_estimator = None) -> StackingRegressor:
    """Stacking集成模型训练"""
    if base_models is None:
        base_models = [
            ('knn', KNeighborsRegressor(n_neighbors=5)),
            ('adaboost', AdaBoostRegressor(n_estimators=1000)),
            ('xgboost', XGBRegressor(n_estimators=1000, max_depth=300))
        ]
    if final_estimator is None:
        final_estimator = LinearRegression()
        
    model = StackingRegressor(estimators=base_models,
                             final_estimator=final_estimator)
    model.fit(X, y.ravel())
    return model

def Stacking_predict(model: StackingRegressor, 
                    X: np.ndarray) -> np.ndarray:
    """Stacking集成模型预测"""
    return model.predict(X).reshape(-1, 1)

# 3. One Layer MLP
def MLP_train(X: np.ndarray, y: np.ndarray,
             hidden_size: int = 50,
             max_iter: int = 1000) -> MLPRegressor:
    """单层MLP训练"""
    model = MLPRegressor(hidden_layer_sizes=(hidden_size,),
                        max_iter=max_iter,
                        random_state=0)
    model.fit(X, y.ravel())
    return model

def MLP_predict(model: MLPRegressor, 
               X: np.ndarray) -> np.ndarray:
    """单层MLP预测"""
    return model.predict(X).reshape(-1, 1)