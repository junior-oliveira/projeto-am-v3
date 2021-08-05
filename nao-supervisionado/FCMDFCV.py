import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean
from warnings import warn


def degrowth(l):
    lo = l.copy()
    l.sort(reverse = True)
    if lo != l:
       warn('Warning: The objective function value did not decrease totally')

def random_U(shape, random_state = None):
    
    if type(shape) == int:
        np.random.seed(random_state)
        h = np.random.random(shape)
        return h/h.sum()
    else:
        np.random.seed(random_state)
        h = np.random.random(shape)
        return h/h.sum(axis= 1).reshape(-1,1) 
        

def euclidean_distance_square_matrices(X,G):
    """
    X is a matrix nxp.
    G is a matrix cxp.
    """
    n,p = X.shape
    c = G.shape[0]
    d2Mi = np.zeros((c,n,p))
    xflat = X.ravel()
    for i in range(c):
        g = np.tile(G[i],n)
        euclidian_dist = (xflat - g)**2
        d2Mi[i] = euclidian_dist.reshape(n,p)
    return d2Mi.astype('float64')
        
def adaptive_quadratic_distance(sqeuclidean, weights):
    """
    sqeuclidean is an array with c matrices nxp related to the euclidean distance square for each cluster.
    weights is a matrix cxp related to the weigths of the clusters for each variable.
    """
    c,n,p = sqeuclidean.shape
    aqd = np.zeros((c,n,p))
    for i in range(c):
        aqd[i] = sqeuclidean[i]*weights[i]
    return aqd.astype('float64')    
        
def membership_degree(aqd, m, lowest_denominator, random_state):
    """
    aqd is an array with c matrices nxp related to the adaptive quadratic distance for each cluster.
    m is the fuzziness parameter
    """
    c,n,p = aqd.shape
    sumj = aqd.sum(axis = 2)
    uki = np.zeros((n,c))
    for k in range(n):
        
        infok = sumj[:,k]
        idx = np.where(infok < lowest_denominator)[0]
        nzeros = len(idx)

        if nzeros == 1:
            ui = np.zeros_like(infok)
            ui[idx] = 1
            uki[k] = ui
            continue
        elif nzeros > 1:
            ui = np.zeros_like(infok)
            ui[idx] = random_U(nzeros, random_state)
            uki[k] = ui
            continue
        else:
            for i in range(c):
                ratio = (infok[i]/infok)**( 1/(m-1) )
                uki[k,i] =  ratio.sum() ** (-1)

    return uki

def prototypes(X,uki,m, G_curr, lowest_denominator):
    """
    X is a matrix nxp related to the dataset;
    uki is a matrix nxc related to the membership degree;
    m is the fuzziness parameter;
    G_curr ie the prototypes's  matrix to be updated
    """
    U = uki ** m
    c,p = G_curr.shape
    for i in range(c):
        
        if (U[:,i]).sum() > lowest_denominator:
            G_curr[i] = np.average(a = X, axis = 0, weights = U[:,i] )
        
    return G_curr
    

def variables_weights(sqeuclidean, uki, m, W_curr, lowest_denominator):
    """
    sqeuclidian is a array with c matrices nxp related to the euclidian distance square for each cluster
    uki is a matrix nxc related to the membership degree;
    m is the fuzziness parameter;
    """
    U = uki ** m
    c = uki.shape[1]
    for i in range(c):
        
        sqi = (sqeuclidean[i] * ( U[:,i].reshape(-1,1) ) ).sum(axis = 0)
        id_lower = np.where(sqi < lowest_denominator)[0]
        
        if len(id_lower) == 0:
            W_curr[i,] = gmean(sqi)/sqi     
    return W_curr


def FCM_DFCV(X, c, m, random_state = 1, T = 150, epsilon = 10 ** (-10), lowest_denominator = 10**(-100)):
    
    names = X.columns
    X = X.to_numpy(dtype = 'float64')

    ######## Inicialization step ########
    n, p = X.shape
    X_unique = np.unique(X,axis = 0)
    nu = len(X_unique)
    Ksort = np.random.choice(nu, c, replace = False)
    G = X_unique[Ksort] # initial prototypes
    U = random_U((n,c),random_state)   # initial memberships degree
    W = np.ones((c,p), dtype = 'float64') # initial weights
    
    D = euclidean_distance_square_matrices(X,G)
    Dw = adaptive_quadratic_distance(D, W)
    
    # Computing the objective function
    d2mi = Dw.sum(axis = 2)
    Jcurr = (((U.T)**m) * d2mi).sum()

    Jlist = [Jcurr.tolist()]
    
    # Iterative step
    t = 0
    while True:
        G = prototypes(X,U,m, G, lowest_denominator)
        D = euclidean_distance_square_matrices(X,G)
        W = variables_weights(D, U, m, W, lowest_denominator)
        Dw = adaptive_quadratic_distance(D, W)
        U = membership_degree(Dw,m,lowest_denominator, random_state + t)
            
        Jprev = Jcurr

        ## Computing the objective function
        #d2mi = Dw.sum(axis = 2)
        #Jcurr = (((U.T)**m) * d2mi).sum()

        Jcurr = 0
        for k in range(n):
            for i in range(c):
                for j in range(p):
                    Jcurr = Jcurr + (U[k,i] ** m) * Dw[i,k,j]

        Jlist.append(Jcurr.tolist())
        t += 1
        if t == T or np.absolute(Jcurr - Jprev) <= epsilon:
            break
            
        # organize output
        cluster_names = list()
        for i in range(c):
            cluster_names.append( 'i = ' + str(i))

    degrowth(Jlist) # Warning in case of the J don't decrease totally 
    G = pd.DataFrame(G, index = cluster_names, columns = names)
    U = pd.DataFrame(U, columns = cluster_names)
    W = pd.DataFrame(W, index = cluster_names, columns = names)
    return {'G': G, 'U':U, 'W':W, 'Jlist':Jlist, 'Jvalue': Jcurr, 't':t }