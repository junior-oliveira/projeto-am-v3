from numpy.lib.function_base import average
import pandas as pd
import numpy as np

from modelos.classificador_bayesiano_parzen import ClassificadorBayesianoParzen
from modelos.classificador_bayesiano import ClassificadorBayesiano
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score
from statistics import mode
import time
from utils import utils
from configuracoes import conf
from imblearn.over_sampling import SMOTE

start_time = time.time()

# Importando a base de dados
dados = utils.get_base_de_dados()

# Rótulo da coluna com as classes
classes = conf.ROTULOS_ATRIBUTOS[len(conf.ROTULOS_ATRIBUTOS)-1]
y = dados[classes]
y = dados['SEQUENCE NAME']

x = dados.drop([classes], axis=1)

# Descomentar para realizar a validação cruzada com normalização
#x = utils.normalizar_dados(x)

# Descomentar para realizar a validação cruzada sem o atributo com correlação significativa
#x = x.drop('GVH', axis=1)


def hyperparameter_tuning(model, x_train, y_train, x_valid, y_valid, hyper_set):

    n = len(hyper_set)
    rs = np.zeros((n,4))

    if model == 'KNN':
        for i in range(n):
            k = hyper_set[i]
            clf = KNeighborsClassifier(n_neighbors = k).fit(x_train, y_train)
            pred = clf.predict(x_valid)

            rs[i,0] = accuracy_score(y_valid, pred)
            rs[i,1] = precision_score(y_valid, pred, average = 'macro' )
            rs[i,2] = recall_score(y_valid, pred, average = 'macro')
            rs[i,3] = f1_score(y_valid, pred, average = 'macro')
    elif model == 'PARZEN':
        for i in range(n):
            h = hyper_set[i]
            clf = ClassificadorBayesianoParzen(h = h).fit(x_train, y_train)
            pred = clf.predict(x_valid)

            rs[i,0] = accuracy_score(y_valid, pred)
            rs[i,1] = precision_score(y_valid, pred, average = 'macro' )
            rs[i,2] = recall_score(y_valid, pred, average = 'macro')
            rs[i,3] = f1_score(y_valid, pred, average = 'macro')
    else:
        return('Error: choose another classifier')

    idm = rs.argmax(axis = 0)
    id_best = mode(idm)
    return hyper_set[id_best]


er_matrix = pd.DataFrame(np.zeros((50,5)),columns = ['knn', 'parzen','bayesian','logistic','ensemble']) # Error classification 
prc_matrix = pd.DataFrame(np.zeros((50,5)),columns = ['knn', 'parzen','bayesian','logistic','ensemble']) # precision
rcl_matrix = pd.DataFrame(np.zeros((50,5)),columns = ['knn', 'parzen','bayesian','logistic','ensemble']) # recall
fms_matrix = pd.DataFrame(np.zeros((50,5)),columns = ['knn', 'parzen','bayesian','logistic','ensemble']) # fmeaseure

# Set of hyperparameters for tuning
kset = np.arange(1,38,2)
hset = np.linspace(0.001, 1, num=19)

best_hyper = pd.DataFrame(np.zeros((50,2)), columns=['k','h'])

rkf = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 10, random_state=2601)
i = 0
for train_index, test_index in rkf.split(x,y):

    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Descomentar para realizar o balanceamento das classes, pela classe minoritária
    #smote = SMOTE(k_neighbors=2, sampling_strategy='minority')
    #x_train, y_train = smote.fit_resample(x_train, y_train)

    ###  tuning of parameters for KNN and Parzen
    x_treino, x_valid , y_treino, y_valid = train_test_split(x_train,y_train,test_size=0.20, 
    random_state = 2601 + i, shuffle = True, stratify = y_train)
    k_best = hyperparameter_tuning('KNN', x_treino, y_treino, x_valid, y_valid, kset ) # KNN
    h_best = hyperparameter_tuning('PARZEN', x_treino, y_treino, x_valid, y_valid, hset ) # PARZEN
    best_hyper.iloc[i,0] = k_best
    best_hyper.iloc[i,1] = h_best


    # KNN
    knn_clf = KNeighborsClassifier(n_neighbors = k_best)
    knn = knn_clf.fit(x_train, y_train)
    pred_knn = knn.predict(x_test)
    er_matrix.iloc[i,0] = 1 - accuracy_score(y_test, pred_knn)
    prc_matrix.iloc[i,0] = precision_score(y_test, pred_knn, average = 'macro' )
    rcl_matrix.iloc[i,0] = recall_score(y_test, pred_knn, average = 'macro')
    fms_matrix.iloc[i,0] = f1_score(y_test, pred_knn, average = 'macro')

    # Parzen
    prz_clf = ClassificadorBayesianoParzen(h = h_best)
    prz = prz_clf.fit(x_train, y_train)
    pred_prz = prz.predict(x_test)
    er_matrix.iloc[i,1] = 1 - accuracy_score(y_test, pred_prz)
    prc_matrix.iloc[i,1] = precision_score(y_test, pred_prz, average = 'macro' )
    rcl_matrix.iloc[i,1] = recall_score(y_test, pred_prz, average = 'macro')
    fms_matrix.iloc[i,1] = f1_score(y_test, pred_prz, average = 'macro')

    #  Bayesian Classifier
    bys_clf = ClassificadorBayesiano()
    bys = bys_clf.fit(x_train, y_train)
    pred_bys = bys.predict(x_test)
    er_matrix.iloc[i,2] = 1 - accuracy_score(y_test, pred_bys)
    prc_matrix.iloc[i,2] = precision_score(y_test, pred_bys, average = 'macro' )
    rcl_matrix.iloc[i,2] = recall_score(y_test, pred_bys, average = 'macro')
    fms_matrix.iloc[i,2] = f1_score(y_test, pred_bys, average = 'macro')

    # logistic regression
    lgr_clf = LogisticRegression(multi_class = 'ovr')
    lgr = lgr_clf.fit(x_train, y_train)
    pred_lgr = lgr.predict(x_test)
    er_matrix.iloc[i,3] = 1 - accuracy_score(y_test, pred_lgr)
    prc_matrix.iloc[i,3] = precision_score(y_test, pred_lgr, average = 'macro' )
    rcl_matrix.iloc[i,3] = recall_score(y_test, pred_lgr, average = 'macro')
    fms_matrix.iloc[i,3] = f1_score(y_test, pred_lgr, average = 'macro')

    #  ensemble
    ens = VotingClassifier(estimators = [('knn',knn_clf), ('parzen',prz_clf), ('bayesian', bys_clf), ('logistic',lgr_clf)], voting ='hard').fit(x_train, y_train)
    pred_ens = ens.predict(x_test)
    er_matrix.iloc[i,4] = 1 - accuracy_score(y_test, pred_ens)
    prc_matrix.iloc[i,4] = precision_score(y_test, pred_ens, average = 'macro' )
    rcl_matrix.iloc[i,4] = recall_score(y_test, pred_ens, average = 'macro')
    fms_matrix.iloc[i,4] = f1_score(y_test, pred_ens, average = 'macro')

    print(i)
    i += 1
er_matrix.to_csv('error_classification_normalized_only.csv',header = True, index = False)
prc_matrix.to_csv('precision_normalized_only.csv',header = True, index = False)
rcl_matrix.to_csv('recall_normalized_only.csv',header = True, index = False) 
fms_matrix.to_csv('fmeasure_normalized_only.csv',header = True, index = False) 
best_hyper.to_csv('best_hyper_normalized_only.csv', header = True, index = False)

print(" %.2f seconds " % ( time.time() - start_time ) )