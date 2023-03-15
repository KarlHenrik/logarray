import numpy as np

def signed_log(x):
    log_values = np.log(np.abs(x))
    signs = np.where(x > 0, 1, -1)

    if np.all(signs == 1):
        return log_values, 1
    if np.all(signs == -1):
        return log_values, -1
    else:
        return log_values, signs

def sub_log_space(a, signa, b, signb):
    return add_log_space(a, signa, b, -1 * signb)

def add_log_space(a, signa, b, signb):
    if np.all(signa == signb):
        return add_log_space_simple(a, b), signa
    else:
        return add_log_space_mixed(a, signa, b, signb)

def add_log_space_simple(a, b):
    # a and b have the same sign
    return np.logaddexp(a, b)

def add_log_space_mixed(a, signa, b, signb):
    sign_eq = signa == signb
    a_big = a > b
    signout = np.where(a_big, signa, signb)

    out = np.empty_like(a)
    # Where same sign
    np.logaddexp(a, b, where = sign_eq, out=out)

    # Where opposite sign
    # Similar to https://github.com/tensorflow/probability/blob/v0.19.0/tensorflow_probability/python/math/generic.py#L652-L681
    np.log1p(-np.exp(-np.abs(b - a)), where = signa!=signb, out=out)
    
    np.add(a, out, where = np.logical_and(a_big, ~sign_eq), out=out) #a is bigger
    np.add(b, out, where = np.logical_and(~a_big, ~sign_eq), out=out) # b is bigger
    
    return out, signout

def mul_log_space(a, signa, b, signb):
    return a + b, signa * signb

def div_log_space(a, signa, b, signb):
    return a - b, signa * signb