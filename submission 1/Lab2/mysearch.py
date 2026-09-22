# Lab 2 and Lab 3 search routines module.
# This file collects the root-finding and minimum-search helpers used across
# the lab scripts so they can be imported from one place.
import numpy as np


def sign(value):
    return np.sign(value)


def err_f(a, b):
    return np.abs(b - a)


def bisection(f, a, b, tol):
    # Keep shrinking the interval until the endpoints are close enough.
    while err_f(a, b) > tol:
        s = np.sign(f(a)) + np.sign(f(b))
        if s == 0 and np.abs(sign(f(a))) > 0 and np.abs(sign(f(b))) > 0:
            c = (a + b) / 2
            if sign(f(c)) == 0:
                a = c
                b = c
            elif sign(f(c)) == sign(f(a)):
                a = c
            elif sign(f(c)) == sign(f(b)):
                b = c
        else:
            print("error, wrong start guesses no root in the interval")
            return None
    if err_f(a, b) < tol:
        return (a + b) / 2


# Secant method helper and root finder.
def sechant_meth(x_1, x_2, f):
    return x_1 - f(x_1) * (x_1 - x_2) / (f(x_1) - f(x_2))


def secant_root(f, x_1, x_2, tol=1e-10, maxit=100):
    for _ in range(maxit):
        x = sechant_meth(x_1, x_2, f)
        if abs(x - x_2) < tol:
            return x
        x_1, x_2 = x_2, x
    return None


def brack(f, a, b, n):
    # Scan an interval and store sub-intervals where the function changes sign.
    brackets = []
    d = np.abs(b - a) / n
    x_1 = a
    x_2 = a + d
    while x_2 <= b:
        s = sign(f(x_1)) + sign(f(x_2))
        if s == 0:
            brackets.append([x_1, x_2])
            x_1 = x_2
            x_2 += d
        else:
            x_1 = x_2
            x_2 += d
        if x_2 > b:
            return np.array(brackets)


def minbracket(f, a, b, n):
    # Find candidate intervals where the middle point is lower than both neighbors.
    minimas = []
    dx = abs(a - b) / n
    x_1 = a
    c = a + dx
    x_2 = a + 2 * dx
    while x_2 <= b:
        if f(x_1) > f(c) and f(x_2) > f(c):
            minimas.append([x_1, x_2])
            x_1 += 2 * dx
            c += 2 * dx
            x_2 += 2 * dx
        else:
            x_1 += dx
            c += dx
            x_2 += dx
    return minimas


def golden_search(f, a, b, tol):
    # Golden-section search for a single minimum inside [a, b].
    phi = (1 + np.sqrt(5)) / 2
    while abs(b - a) > tol:
        c = b - (b - a) / phi
        d = a + (b - a) / phi
        if f(c) < f(d):
            b = d
        if f(d) < f(c):
            a = c
    return (a + b) / 2


def minimum_finder(f, a, b, n, tol):
    # First bracket all candidate minima, then refine each one with golden search.
    brackets = minbracket(f, a, b, n)
    return [golden_search(f, x_min, x_max, tol) for x_min, x_max in brackets]
