'''# 高斯积分
- gausf(f, a=0, b=1, n=10)
- gausf2(f, xa=0, xb=1, ya=0, yb=1, n=10)
- gausf3(f, xa=0, xb=1, ya=0, yb=1, za=0, zb=1, n=10)
'''

import numpy as np

# 读取高斯积分的系数表
def read_gauss_data() -> dict:
    '''## 读取高斯系数数据
    `return`: `coefficients_all` : `dict`
    '''
    fi=open('Const_gauss.txt', 'r')
    s=fi.read()
    xyall=s.split('\n')
    fi.close()

    coefficients_all={}

    for n in range(1, round(len(xyall)/3)):
        x=xyall[3*n-3].split()
        Xs = np.asarray(x, dtype=np.float64)
        w=xyall[3*n-2].split()
        Ws = np.asarray(w, dtype=np.float64)

        if len(x)==n and len(w)==n:
            coefficients_all.update({n:np.array([Xs,Ws])})

    return coefficients_all

coefficients_all = read_gauss_data()


# 通用一维定积分
def gausf(f, a=0, b=1, n=10):
    '''## 通用的一维高斯定积分
    `input`:    函数 `f`, 积分边界 `a`, `b`，精度 `n`
    `return`:   定积分结果
    '''
    if n not in coefficients_all:
        print('not a correct n')
        return False
    
    xs = (a+b)/2+(b-a)/2*coefficients_all[n][0]
    result=(coefficients_all[n][1]*f(xs)).sum()

    return result*(b-a)/2


# 通用的二维高斯定积分
def gausf2(f, xa=0, xb=1, ya=0, yb=1, n=10):
    '''## 通用的二维高斯定积分
    `input`:    函数 `f`, 积分边界 `xa`, `xb`, `ya`, `yb`，精度 `n`
    `return`:   定积分结果
    '''
    if n not in coefficients_all:
        print('not a correct n')
        return None

    xs = (xa+xb)/2+(xb-xa)/2*coefficients_all[n][0]
    ys = (ya+yb)/2+(yb-ya)/2*coefficients_all[n][0]
    wx = coefficients_all[n][1]*(xb-xa)/2
    wy = coefficients_all[n][1]*(yb-ya)/2

    xys = np.meshgrid(xs, ys)   # 取点
    wws = np.meshgrid(wx, wy)   # 权重
    result = (wws[0] *wws[1] *f(xys[0], xys[1])).sum()

    return result


# 通用的三维高斯定积分
def gausf3(f, xa=0, xb=1, ya=0, yb=1, za=0, zb=1, n=10):
    '''## 通用的三维高斯定积分
    `input`:    函数 `f`, 积分边界 `xa`, `xb`, `ya`, `yb`, `za`, `zb`, 精度 `n`
    `return`:   定积分结果
    '''
    if n not in coefficients_all:
        print('not a correct n')
        return None

    xs = (xa+xb)/2+(xb-xa)/2*coefficients_all[n][0]
    ys = (ya+yb)/2+(yb-ya)/2*coefficients_all[n][0]
    zs = (za+zb)/2+(zb-za)/2*coefficients_all[n][0]
    wx = coefficients_all[n][1]*(xb-xa)/2
    wy = coefficients_all[n][1]*(yb-ya)/2
    wz = coefficients_all[n][1]*(zb-za)/2

    xyzs = np.meshgrid(xs, ys, zs)   # 取点
    wwws = np.meshgrid(wx, wy, wz)   # 权重
    result = (wwws[0] *wwws[1] *wwws[2] *f(xyzs[0], xyzs[1], xyzs[2])).sum()

    return result
