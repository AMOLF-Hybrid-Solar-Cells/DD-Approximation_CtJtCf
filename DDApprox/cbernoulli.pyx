from libc.math cimport exp

# Bernoulli constants
cdef float x1 = -29.98554969583478
cdef float x2 = -6.623333247568475e-05
cdef float x3 = 9.273929305608775e-06
cdef float x4 = 28.937746930131606
cdef float x5 = 740.5388178707861

def bernoulli(float x):
    if x <= x1:
        return -x
    elif x > x1 and x < x2:
        return x/(exp(x)-1)
    elif x >= x2 and x <= x3:
        return 1-x/2
    elif x > x3 and x < x4:
        return x*exp(-x)/(1-exp(-x))
    elif x >= x4 and x < x5:
        return x*exp(-x)
    elif x>=x5:
        return 0

def diff_bernoulli(float x):
    cdef float b = bernoulli(x)
    if x >= x2 and x <= x3:
        return (x-2)/4
    else:
        return b/x*(1-b-x)