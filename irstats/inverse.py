import scipy.stats as st

def t_inv(phi, P):
    return st.t.ppf(1 - (1 - P) / 2, phi)

def F_inv(phi1, phi2, P):
    return st.f.ppf(1 - P, phi1, phi2)
