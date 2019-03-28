import scipy.stats as st

def t_inv(phi, P):
    return st.t.ppf(1 - (1 - P) / 2, phi)
