from utils import utils
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

dados = utils.get_base_de_dados()
y = dados['SEQUENCE NAME']
x = dados.drop('SEQUENCE NAME', axis=1)
x_treino, x_teste , y_treino, y_teste = train_test_split(x,y,test_size=0.3, stratify=y, random_state=15)

def modelos (a, b):
    from classificador_bayesiano import ClassificadorBayesiano
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import VotingClassifier
    

    x = a
    y = b

    kfold = KFold(n_splits=10)

    bayes_clf = ClassificadorBayesiano()
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    reg_log_clf = LogisticRegression(max_iter=200)
    arv_dec_clf = DecisionTreeClassifier()
    ensemble = VotingClassifier(estimators=[('bayes_clf', bayes_clf), ('knn_clf', knn_clf),('reg_log_clf', reg_log_clf), ('arv_dec_clf', arv_dec_clf)], voting='hard')


    bayes_clf_cv = cross_val_score(bayes_clf, x, y, cv = kfold).mean()
    knn_clf_cv = cross_val_score(knn_clf, x, y, cv = kfold).mean()
    reg_log_clf_cv = cross_val_score(reg_log_clf, x, y, cv = kfold).mean()
    arv_dec_clf_cv = cross_val_score(arv_dec_clf, x, y, cv = kfold).mean()
    ensemble_cv = cross_val_score(ensemble, x, y, cv = kfold).mean()

    dic_modelos = {'Bayes':bayes_clf_cv, 'KNN':knn_clf_cv, 'Regressao':reg_log_clf_cv, 'Arvore':arv_dec_clf_cv, 'Ensemble':ensemble_cv}
    melhor_modelo = max(dic_modelos, key=dic_modelos.get)

    print('Bayes', bayes_clf_cv, '\nKNN', knn_clf_cv, '\nRegressao', reg_log_clf_cv, '\nArvore', arv_dec_clf_cv, '\nEnsemble', ensemble_cv)
    print('\nO melhor modelo é:', melhor_modelo)
    print('Com valor de:', dic_modelos[melhor_modelo])

modelos(x, y)