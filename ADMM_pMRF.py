# Terry 2024-12-16
# 实现ADMM求解pMRF

import numpy as np 
import cvxpy as cp
# import numpy.linalg as la 


def penalty_scad_i(alpha:float,lamb:float,w:float) -> float:
    """
    计算scad对单个w的惩罚
    : alpha 惩罚的参数
    : lamb 惩罚的系数
    : w 单个w
    """
    w = abs(w)
    if w < lamb:
        ans = lamb*w
    elif w < alpha*lamb:
        ans = (alpha*lamb*w - (w**2 + lamb**2)/2)/(alpha-1)
    else:
        ans = (alpha+1)*lamb**2/2
    return ans 

def penalty_mcp_i(alpha:float,lamb:float,w:float) -> float:
    """
    计算mcp对单个w的惩罚
    : alpha 惩罚的参数
    : lamb 惩罚的系数
    : w 单个w
    """
    w = abs(w)
    if w < alpha*lamb:
        ans = lamb*w - w**2/(2*alpha)
    else:
        ans = 0.5*alpha*lamb**2
    return ans 

def penalty_tlp_i(theta:float,lamb:float,w:float) -> float:
    """
    计算tlp对单个w的惩罚
    : theta 惩罚的参数
    : lamb 惩罚的系数
    : w 单个w
    """
    w = abs(w)
    ans = lamb * min(w,theta)
    return ans 


def Penalty(alpha:float,theta:float,lamb:float,w:list[float],method:str) -> float:
    """
    计算惩罚
    : alpha 惩罚的参数 scad 和 mcp中需要 
    : theta 惩罚的参数 tlp 中需要
    : lamb 惩罚的系数
    : w 权重list
    : method 惩罚的方法
    """
    if method == "SCAD":
        return penalty_scad(alpha,lamb,w)
    elif method == "MCP":
        return penalty_mcp(alpha,lamb,w)
    elif method == "TLP":
        return penalty_tlp(theta,lamb,w)

def penalty_scad(alpha:float,lamb:float,w:list[float]) -> float:
    """
    计算scad对w的惩罚
    : alpha 惩罚的参数
    : lamb 惩罚的系数
    : w 权重list
    """
    ans = 0
    for wi in w:
        tmp = penalty_scad_i(alpha,lamb,wi)
        ans+=tmp
    return ans 

def penalty_mcp(alpha:float,lamb:float,w:list[float]) -> float:
    """
    计算mcp对w的惩罚
    : alpha 惩罚的参数
    : lamb 惩罚的系数
    : w 权重list
    """
    ans = 0
    for wi in w:
        tmp = penalty_mcp_i(alpha, lamb, wi)
        ans+=tmp
    return ans 

def penalty_tlp(theta:float,lamb:float,w:list[float]) -> float:
    """
    计算tlp对w的惩罚
    : theta 惩罚的参数
    : lamb 惩罚的系数
    : w 权重list    
    """
    ans = 0
    for wi in w:
        tmp = penalty_tlp_i(theta, lamb, wi)
        ans+=tmp
    return ans 



def close_form(lamb:float,x1i:float,z1i:float,x2i:float,z2i:float,beta:float,method:str) -> float:
    
    # 固定死的参数 注意和其他地方的hard code 对齐
    alpha = 3.7 
    theta = 0.05
    
    phi = (z1i + x1i/beta + z2i + x2i/beta)/2
    beta = 2*beta 
    
    if method == "SCAD":
        
        w1 = np.sign(phi)* min(lamb,max(0,abs(phi) - lamb/beta))
        tmp2 = (beta*abs(phi)*(theta-1) - lamb*theta)/(beta*(theta-2))
        w2 = np.sign(phi)*min(lamb*theta,max(lamb,tmp2))
        w3 = np.sign(phi)*max(lamb*theta,abs(phi))
        
        w_list  = [w1,w2,w3]
        h_list = []
        for w in w_list :
            hi = 0.5*(w - phi)**2 +  penalty_scad_i(alpha,lamb,w)/beta
            h_list.append(hi)
        
        # print(h_list)
        # print(x_list)
        idx = np.argmin(h_list)
        ans = w_list[idx]
    
    elif method == 'MCP':
        tmp = theta*(beta*abs(phi) - lamb)/(beta*(theta-1))
        z_list  = [0,theta*lamb,min(theta*lamb,max(0,tmp))]
        tmp2_list = []
        for zi in z_list :
            tmp2 = 0.5*(zi - abs(phi))**2 + lamb/beta*zi - zi**2/(2*theta)
            tmp2_list.append(tmp2)
        m = z_list[np.argmin(tmp2_list)]
        w1 = np.sign(phi)*m
        w2 = np.sign(phi)*max(theta*lamb,abs(phi))
        
        h1 = 0.5*(w1 - phi)**2 +  penalty_mcp_i(alpha,lamb,w1)/beta
        h2 = 0.5*(w2 - phi)**2 +  penalty_mcp_i(alpha,lamb,w2)/beta
        
        if h1 < h2 :
            ans = w1 
        else:
            ans = w2
            
    elif method == "TLP":
        w1 = np.sign(phi) * max(theta,abs(phi))
        w2 = np.sign(phi) * min(theta,max(0,abs(phi)-lamb/beta))
        
        h1 = 0.5*(w1 - phi)**2 +  penalty_tlp_i(theta,lamb,w1)/beta
        h2 = 0.5*(w2 - phi)**2 +  penalty_tlp_i(theta,lamb,w2)/beta
        
        if h1 < h2 :
            ans = w1 
        else:
            ans = w2
    else:
        raise ValueError("method not supported")
    return(ans)



def cp_solver_1(y:np.ndarray,y_preds:np.ndarray,km:np.ndarray,sigma2_mean:float,beta:float,w:np.ndarray,x1:np.ndarray) -> np.ndarray:
    """
    式3.12的cp solver
    """
    n,M = np.shape(y_preds)

    # 处理shape
    km = km.ravel() 
    y = y.ravel()
    w = w.ravel()
    x1 = x1.ravel()

    z = cp.Variable(M)

    y_hat = y_preds@z 
    # print(y_hat.shape)
    cost = 1/(2*n) * cp.sum_squares(y - y_hat) + 1/n * sigma2_mean * cp.sum(km @ z) + beta/2 * cp.sum_squares(w - (z + x1/beta))

    myprob = cp.Problem(cp.Minimize(cost), [cp.sum(z) == 1, z >= 0])
    myprob.solve()
    # print("MRF的解: ", prob.value)
    # print("z的解: ", z.value)

    # 特别小的先给压成0
    z1 = z.value
    z1[abs(z1)<1e-10] = 0 
    z1 = z1/sum(z1)
    return (z1)


def cp_solver_2(y:np.ndarray,y_preds:np.ndarray,km:np.ndarray,sigma2_mean:float,beta:float,w:np.ndarray,x2:np.ndarray) -> np.ndarray:
    """
    式3.13的cp solver
    """

    n,M = np.shape(y_preds)

    # 处理shape
    km = km.ravel() 
    y = y.ravel()
    w = w.ravel()
    x2 = x2.ravel()

    z = cp.Variable(M)

    # cost 和 cp_solver-1 中的不一样
    cost = beta/2 * cp.sum_squares(w - (z + x2/beta))

    myprob = cp.Problem(cp.Minimize(cost), [cp.sum(z) == 1, z >= 0])
    myprob.solve()

    # 特别小的先给压成0
    z2 = z.value
    z2[abs(z2)<1e-10] = 0 
    z2 = z2/sum(z2)
    return (z2)

