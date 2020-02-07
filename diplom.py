import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import copy

a = 0.03
eps0 = 4
D = 0.1
dx = 0.01
beta = []
kritdelt = []

def DR(a, b, eps0, delt, x):
    return a*(1-(sqrt(x)-b)**2)*(1-x)-x*(1-(sqrt(x)-b)**2)*exp(-eps0*(2*x+delt))+D*((1-x)-(1-(sqrt(x)-b)**2))


def virazenie():
    a, b, m, x, eps0, delta, D = symbols('a,b,m,x,eps0,delta,D')
    R = a * (1 - (sqrt(x) - b) ** 2) * (1 - x) - x * (1 - (sqrt(x) - b) ** 2) * exp(-eps0 * (2 * x + delta)) + D * (
                (1 - x) - (1 - (sqrt(x) - b) ** 2))
    M = solve(R, delta)
    d = diff(R, x)
    G = a * (1 - x) * (1 - (sqrt(x) - b) ** 2) - x * (1 - (sqrt(x) - b) ** 2) * exp(
        -2 * eps0 * x - delta * eps0) + D * ((sqrt(x) - b) ** 2 - x)
    N = solve(G, x)

    V = -a * ((1 / 3) * x ** 3 - (1 / 2) * x ** 2 + 2 * b * (-(2 / 5) * x ** (5 / 2) + (2 / 3) * x ** (3 / 2)) + (
                -b ** 2 + 1) * (x - (1 / 2) * x ** 2)) - (2 * (-(1 / 4) * x ** 2 * exp(-2 * eps0 * x) / eps0 + (
                -(1 / 4) * x * exp(-2 * eps0 * x) / eps0 - (1 / 8) * exp(-2 * eps0 * x) / eps0 ** 2) / eps0)) / exp(
        eps0 * delta) + 4 * b * (-(1 / 4) * x ** (3 / 2) * exp(-2 * eps0 * x) / eps0 + (3 / 4) * (
                -(1 / 4) * sqrt(x) * exp(-2 * eps0 * x) / eps0 + (1 / 16) * sqrt(pi) * sqrt(2) * erf(
            sqrt(2) * sqrt(eps0) * sqrt(x)) / eps0 ** (3 / 2)) / eps0) / exp(eps0 * delta) - 2 * b ** 2 * (
                    -(1 / 4) * x * exp(-2 * eps0 * x) / eps0 - (1 / 8) * exp(-2 * eps0 * x) / eps0 ** 2) / exp(
        eps0 * delta) + (
                    2 * (-(1 / 4) * x * exp(-2 * eps0 * x) / eps0 - (1 / 8) * exp(-2 * eps0 * x) / eps0 ** 2)) / exp(
        eps0 * delta) - D * (-(4 / 3) * b * x ** (3 / 2) + b ** 2 * x)

