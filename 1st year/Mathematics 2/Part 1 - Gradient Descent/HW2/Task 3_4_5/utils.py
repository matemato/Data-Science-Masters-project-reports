import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

plt.style.use('seaborn-darkgrid')

def f1(x):
    # real min at f(-0.166667, -0.229167, 0.166667) = -0.197917
    x, y, z = x
    return (x - z) ** 2 + (2 * y + z) ** 2 + (4 * x - 2 * y + z) ** 2 + x + y

def f2(x):
    x, y, z = x
    return (x - 1) ** 2 + (y - 1) ** 2 + 100 * (y - x ** 2) ** 2 + 100 * (z - y ** 2) ** 2

def f3(x):
    #real min at f(3, 0.5) = 0
    x, y = x
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

def grad1(x):
    x, y, z = x
    return np.array([34*x-16*y+6*z+1, 
                     -16*x+16*y+1, 
                     6*x+6*z])

def grad2(x):
    x, y, z = x
    return np.array([2*(x-1)-400*x*(y-x**2), 
                     2*(y-1)+200*(y-x**2)-400*y*(z-y**2), 
                     200*(z-y**2)])

def grad3(x):
    x, y = x
    return np.array([-12.75 + 3*y + 4.5*y**2 + 5.25*y**3 + 2*x*(3 - 2*y - y**2 - 2*y**3 + y**4 + y**6),
                     6*x * (0.5 + 1.5 * y + 2.625 * y**2 + x * (-1/3 -1/3 * y - y**2 + 2/3 * y**3 + y**5))])

def hessian1(x):
    x, y, z = x
    return np.array([[34, -16, 6],
                     [-16, 16, 0],
                     [6, 0, 6]])

def hessian2(x):
    x, y, z = x
    return np.array([
        [-400*(y-x**2) + 800*x**2 + 2, -400*x, 0],
        [-400*x, -400*(z-y**2) + 800*y**2 + 202, -400*y],
        [0, -400*y, 200]
    ])

def hessian3(x):
    x, y = x
    return np.array([[2*(y**6+y**4-2*y**3-y**2-2*y+3), 2*x*(6*y**5+4*y**3-6*y**2-2*y-2)+15.75*y**2+9*y+3],
                     [2*x*(6*y**5+4*y**3-6*y**2-2*y-2)+15.75*y**2+9*y+3, 2*x*(x*(15*y**4+6*y**2-6*y-1)+6*2.625*y+4.5*y)]])

def grad(x, f_number):
    if f_number == 1: return grad1(x)
    elif f_number == 2: return grad2(x)
    elif f_number == 3: return grad3(x)
    else: print("Wrong function number for gradient.")

def hessian(x, f_number):
    if f_number == 1: return hessian1(x)
    elif f_number == 2: return hessian2(x)
    elif f_number == 3: return hessian3(x)
    else: print("Wrong function number for Hessian.")

def gd(x, gamma, T, f_number, time_limit=None):
    xi = [x]
    start_time = time()
    for i in range(T):
        x = x - gamma * grad(x, f_number)
        xi.append(x)
        if np.linalg.norm(x - xi[-2]) < 10e-15 or (time_limit is not None and time() - start_time > time_limit):
            return np.array(xi)
    return np.array(xi)

def polyak(x, x_prev, gamma, v, T, f_number, time_limit=None):
    xi = [x]
    start_time = time()
    for i in range(T):
        x = x - gamma * grad(x, f_number) + v*(x-x_prev)
        x_prev = xi[-1]
        xi.append(x)
        if np.linalg.norm(x - xi[-2]) < 10e-15 or (time_limit is not None and time() - start_time > time_limit):
            return np.array(xi)
    return np.array(xi)

def nesterov(x, x_prev, gamma, v, T, f_number, time_limit=None):
    xi = [x]
    start_time = time()
    for i in range(T):
        x = x - gamma * grad(x + v*(x - x_prev), f_number) + v*(x-x_prev)
        x_prev = xi[-1]
        xi.append(x)
        if np.linalg.norm(x - xi[-2]) < 10e-15 or (time_limit is not None and time() - start_time > time_limit):
            return np.array(xi)
    return np.array(xi)

def adagrad(x, gamma, T, f_number, time_limit=None):
    cache = 0
    xi = [x]
    start_time = time()
    for i in range(T):
        cache += (grad(x, f_number)**2).astype(np.float64)
        x = x - (gamma * grad(x, f_number)) / (np.sqrt(cache) + 1e-7)
        xi.append(x)
        if np.linalg.norm(x - xi[-2]) < 10e-15 or (time_limit is not None and time() - start_time > time_limit):
            return np.array(xi)
    return np.array(xi)

def newton(x, T, f_number, time_limit=None):
    xi = [x]
    start_time = time()
    for i in range(T):
        x = x - np.linalg.inv(hessian(x, f_number)) @ grad(x, f_number)
        xi.append(x)
        if np.linalg.norm(x - xi[-2]) < 10e-15 or (time_limit is not None and time() - start_time > time_limit):
            return np.array(xi)
    return np.array(xi)

def BFGS(x, T, f_number, time_limit=None):
    xi = [x]
    B = np.eye(len(x)) / 100000
    start_time = time()
    for i in range(T):
        x = x - B @ grad(x, f_number)
        x_prev = xi[-1]
        delta = np.nan_to_num(x - x_prev)
        gamma = np.nan_to_num(grad(x, f_number) - grad(x_prev, f_number))
        dot = delta @ gamma
        if dot == 0:
            print("BFGS terminating.")
            return np.array(xi) 
        
        B = B - (( np.outer(delta, gamma) @ B + B @ np.outer(gamma, delta) ) / dot) + (1 + (gamma.T @ B @ gamma) / dot) * ((np.outer(delta, delta)) / dot)
        B = np.nan_to_num(B)
        xi.append(x)  
        if np.linalg.norm(x - xi[-2]) < 10e-15 or (time_limit is not None and time() - start_time > time_limit):
            return np.array(xi)      
    return np.array(xi)


def plot_descent(xi1, xi2, x_min, descent, ax, T=None, time_limit=None):
    
    x, y = np.linspace(-5, 5, 100), np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)

    Z = np.log10(f3([X, Y]))

    levels = np.linspace(np.min(Z), np.max(Z), 30)

    ax.set_box_aspect(1)

    ax.contourf(X,Y,Z, levels)

    ax.plot(xi1[:,0], xi1[:,1], '.-r', label=f'{descent} steps')
    ax.plot(xi1[0,0], xi1[0,1], '.g', label=f'start_1 ({xi1[0,0]}, {xi1[0,1]})')

    ax.plot(xi2[:,0], xi2[:,1], '.-r')
    ax.plot(xi2[0,0], xi2[0,1], '.y', label=f'start_2 ({xi2[0,0]}, {xi2[0,1]})')
    
    ax.plot(xi1[-1,0], xi1[-1,1], '.', color='orange', label=f'end_1 ({round(xi1[-1,0], 2)}, {round(xi1[-1,1], 2)}) with {len(xi1)-1} steps')
    ax.plot(xi2[-1,0], xi2[-1,1], '.', color='orange', label=f'end_2 ({round(xi2[-1,0], 2)}, {round(xi2[-1,1], 2)}) with {len(xi2)-1} steps')
    ax.plot(x_min[0],x_min[1], '.b', label=f'minima ({x_min[0]}, {x_min[1]})')

    if T is not None: ax.set_title(f'{descent} with {T} steps.')
    if time_limit is not None: ax.set_title(f'{descent} with time limit: {time_limit}')
    ax.set_ylabel('y')
    ax.set_xlabel('x')

    ax.legend(labelcolor='linecolor')

def plot_descent3D(xi1, xi2, x_min, descent, ax, T=None, time_limit=None):

    ax.scatter(xi1[1:,0], xi1[1:,1], xi1[1:,2], color='orange', label=f'Steps for first point')
    ax.scatter(xi2[:,0], xi2[:,1], xi2[:,2], color='darkred', label=f'Steps for second point')

    ax.scatter(xi1[0,0], xi1[0,1], xi1[0,2], color='yellow', alpha=1, label=f'start_1 ({xi1[0,0]}, {xi1[0,1]}, {xi1[0,2]})')    
    ax.scatter(xi2[0,0], xi2[0,1], xi2[0,1], color='lightgreen', alpha=1, label=f'start_2 ({xi2[0,0]}, {xi2[0,1]}, {xi2[0,2]})')

    ax.scatter(xi1[-1,0], xi1[-1,1], xi1[-1,2], color='red', label=f'end_1 ({round(xi1[-1,0], 2)}, {round(xi1[-1,1], 2)}, {round(xi1[-1,2], 2)}) with {len(xi1)-1} steps')
    ax.scatter(xi2[-1,0], xi2[-1,1], xi2[-1,2], color='green', alpha=1, label=f'end_2 ({round(xi2[-1,0], 2)}, {round(xi2[-1,1], 2)}, {round(xi2[-1,2], 2)}) with {len(xi2)-1} steps')
    ax.scatter(x_min[0], x_min[1], x_min[2], color='blue', alpha=1, label=f'minima ({round(x_min[0], 2)}, {round(x_min[1], 2)}, {round(x_min[2],2)})')

    if T is not None: ax.set_title(f'{descent} with {T} steps.')
    if time_limit is not None: ax.set_title(f'{descent} with time limit: {time_limit}')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_zlabel('z')
    ax.legend(bbox_to_anchor=(0, -0.2, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2)

    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_zlim(-2,2)

def plot_all(x1_1, x1_2, x_min, gammas, v, T, time_limit, optimizing_functions, f, f_num):
    if f_num == 3: fig, axs = plt.subplots(2, 3, sharey=True, figsize=(21,14))
    else: fig, axs = plt.subplots(2, 3, figsize=(21,14), subplot_kw=dict(projection='3d')) ###
    
    print("Difference between final function value and minima function value:\n")
    for i, descent in enumerate(optimizing_functions):
        if descent == 'Gradient descent':
            xi1 = gd(x1_1, gammas[0,i], T, f_num, time_limit)
            xi2 = gd(x1_2, gammas[1,i], T, f_num, time_limit)
        elif descent == 'Polyak GD':
            xi1 = polyak(x1_1, x1_1, gammas[0,i], v, T, f_num, time_limit)
            xi2 = polyak(x1_2, x1_2, gammas[1,i], v, T, f_num, time_limit)
        elif descent == 'Nesterov GD':
            xi1 = nesterov(x1_1, x1_1, gammas[0,i], v, T, f_num, time_limit)
            xi2 = nesterov(x1_2, x1_2, gammas[1,i], v, T, f_num, time_limit)
        elif descent == 'AdaGrad':
            xi1 = adagrad(x1_1, gammas[0,i], T, f_num, time_limit)
            xi2 = adagrad(x1_2, gammas[1,i], T, f_num, time_limit)
        elif descent == 'Newton method':
            xi1 = newton(x1_1, T, f_num, time_limit)
            xi2 = newton(x1_2, T, f_num, time_limit)
        elif descent == 'BFGS':
            xi1 = BFGS(x1_1, T, f_num, time_limit)
            xi2 = BFGS(x1_2, T, f_num, time_limit)
        if f_num == 3: plot_descent(xi1, xi2, x_min, descent, axs[i//3][i%3], T=T, time_limit=time_limit)
        else: plot_descent3D(xi1, xi2, x_min, descent, axs[i//3][i%3], T, time_limit)
        print(f'{descent} for starting point ({xi1[0,0]}, {xi1[0,1]}): {abs(f(xi1[-1]) - f(x_min))}')
        print(f'{descent} for starting point ({xi2[0,0]}, {xi2[0,1]}): {abs(f(xi2[-1]) - f(x_min))}')