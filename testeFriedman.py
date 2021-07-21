from utils import utils
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import numpy as np

dados = utils.get_base_de_dados()

#Transformando o tipo dos dados para numerico
nuc = dados['NUC'].apply(pd.to_numeric)
vac = dados['VAC'].apply(pd.to_numeric)
pox = dados['POX'].apply(pd.to_numeric)
erl = dados['ERL'].apply(pd.to_numeric)
mit = dados['MIT'].apply(pd.to_numeric)
alm = dados['ALM'].apply(pd.to_numeric)
gvh = dados['GVH'].apply(pd.to_numeric)
mcg = dados['MCG'].apply(pd.to_numeric)


#aplicando o teste de friedman em cada coluna
fs = friedmanchisquare(nuc, vac, pox, erl, mit, alm, gvh, mcg)

print()
print(fs)
print()

#aplicando o teste de nemnyi em cada coluna
data = np.array([nuc, vac, pox, erl, mit, alm, gvh, mcg])
nemenyi = sp.posthoc_nemenyi_friedman(data.T)

print()
print(nemenyi)