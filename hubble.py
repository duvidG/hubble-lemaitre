#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 17:44:50 2021

@author: dgiron
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit as cf
import math
from uncertainties import ufloat, unumpy
from uncertainties.umath import *
from uncertainties.unumpy import nominal_values as nv
from uncertainties.unumpy import std_devs as st

M = -21
l_01 = 3933.67
l_02 = 3968.47
c = 300000

def rs(l, l0):
    return (l - l0) / l0

def f(x, a):
    
    return a * x


def ajuste(x, y):
    ppot, pcov = cf(f, nv(x), nv(y))
    perr = np.sqrt(np.diag(pcov))
    
    plt.clf()
    xx = np.linspace(min(nv(x)), max(nv(x)), 1000)
    yy = f(xx, *ppot)
    plt.errorbar(nv(x), nv(y), xerr=st(x), yerr=st(y), fmt='o', label='Data')
    plt.plot(xx, yy, label=r'$v = a \cdot d$')
    plt.xlabel('Distance/Mpc')
    plt.ylabel('v/km/s')
    plt.grid()
    plt.legend()
    # plt.savefig('hubble.png', dpi=720)
    plt.show()
    
    return ufloat(ppot[0], perr[0])

def tabla_latex(tabla, ind, col, r):
    """
    Prints an array in latex format
    Args:
        tabla: array to print in latex format (formed as an array of the variables (arrays) that want to be displayed)
        ind: list with index names
        col: list with columns names
        r: number of decimals to be rounded
        
    Returns:
        ptabla: table with the data in a pandas.Dataframe format
    """
    tabla = tabla.T
    # tabla = tabla.round(r)
    ptabla = pd.DataFrame(tabla, index=ind, columns=col)
    print("Tabla en latex:\n")
    print(ptabla.to_latex(index=False, escape=False))
    return ptabla


def main():
    
    datos = np.genfromtxt('datos.txt', delimiter=',')
    datos = datos[1:]
    datos2 = np.genfromtxt('datos2.CSV', delimiter=',')
    datos2 = datos2[1:]
    
    m = datos[:, 0]
    delta_m = m * 1/datos[:, 1]
    lam_1 = datos[:, 3]
    lam_2 = datos[:, 4]
    
    ### Para quitar la galaxia pocha
    i = False
    if i:
        m = m[:-1]
        delta_m = delta_m[:-1]
        lam_1 = lam_1[:-1]
        lam_2 = lam_2[:-1]
    ###    
        
    m_err = unumpy.uarray(m, delta_m)
    
    dist = 10 ** ((m_err - M + 5) / 5)
    
    rs1 = rs(lam_1, l_01)
    rs2 = rs(lam_2, l_02)

    rs_med = []
    err = []

    for j, i in enumerate(rs1):
        rs_med.append(np.mean([i, rs2[j]]))
        err.append(np.std([i, rs2[j]]))    
    rs_med = np.array(rs_med)
    err = np.array(err)

    rs_err = unumpy.uarray(rs_med, err)

    hubb = ajuste(dist / 10 ** 6, rs_err * c)

    print('H_0 = ', hubb, 'km/s/Mpc')
    
    print('t_H', (1 / hubb) * 3.0869e19 / 3.17098e7 / 10 ** 9, 'Gy')
    
    tab = np.array([datos[:, -1], [-21 for i in datos[:, 1]], datos[:, 1], m, lam_1, lam_2, np.round(rs1, 3), np.round(rs2, 3), rs_err])
    tabla_latex(tab, ['' for i in datos[:,1]], ['$Number$', '$M$', '$SNR$', '$m$', '$\lambda_k/\AA$', '$\lambda_h/\AA$', '$z_k$', '$z_h$', '$<z>$'], 3)
    
    tab = np.array([datos[:, -1], dist/10**6,  rs_err*c])
    tabla_latex(tab, ['' for i in datos[:,1]], ['$Number$', '$d$/Mpc', '$<v>$/km/s'], 3)
    
main()