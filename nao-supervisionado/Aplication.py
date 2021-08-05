from numpy.lib.npyio import load
import pandas as pd
import numpy as np
from FCMDFCV import FCM_DFCV
import time


start_time = time.time()

x = pd.read_csv('yeast.csv')
labels = x['class']
x = x.drop(['class'], axis=1)

M = [1.0001, 1.1, 1.6, 2.0]
c = 10
for m in M:
    seed = 33
    for i in range(100):

        Gname1 = 'Resultados/application1/prototype/G' + str(i) + '(' + str(m) + ').csv'
        Uname1 = 'Resultados/application1/degree/U' + str(i) + '(' + str(m) + ').csv'
        Wname1 = 'Resultados/application1/weight/W' + str(i)   + '(' + str(m) + ').csv'
        Jname1 = 'Resultados/application1/j/J' + '(' + str(m) + ').csv'
        Tname1 = 'Resultados/application1/t/T' + '(' + str(m) + ').csv'
        
        rs1 = FCM_DFCV(X = x, c = c, m = m,random_state= seed + i)

        rs1['G'].to_csv(Gname1, header = True, index = False)
        rs1['U'].to_csv(Uname1, header = True, index = False)
        rs1['W'].to_csv(Wname1, header = True, index = False)
        pd.Series(rs1['Jvalue']).to_csv(Jname1, header = False, index = False, mode = 'a')
        pd.Series(rs1['t']).to_csv(Tname1, header = False, index = False, mode = 'a')

        
        print('i{}({})'.format(i,m))
        
print(" %.2f seconds " % ( time.time() - start_time ) )