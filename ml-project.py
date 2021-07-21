#import time
from utils import utils
from sklearn.model_selection import train_test_split
from classificador_bayesiano import ClassificadorBayesiano
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

# t = time.time()
# ...
# elapsed = time.time() - t
# print('\nTempo gasto: '+str(elapsed))

dados = utils.get_base_de_dados()
y = dados['SEQUENCE NAME']
x = dados.drop('SEQUENCE NAME', axis=1)
x_treino, x_teste , y_treino, y_teste = train_test_split(x,y,test_size=0.3, stratify=y)
NB = ClassificadorBayesiano()
NB.fit(x_treino, y_treino)

# Implementação do classificador bayesiano de acordo com o que foi pedido na disciplina de AM
bayes_clf = ClassificadorBayesiano()
knn_clf = KNeighborsClassifier(n_neighbors=5)
reg_log_clf = LogisticRegression(max_iter=200)
arv_dec_clf = DecisionTreeClassifier()


bayes_clf.fit(x_treino, y_treino)
knn_clf.fit(x_treino, y_treino)
reg_log_clf.fit(x_treino, y_treino)
arv_dec_clf.fit(x_treino, y_treino)


res_bayes_clf = bayes_clf.score(x_teste, y_teste)
res_knn_clf = knn_clf.score(x_teste, y_teste)
res_reg_log_clf = reg_log_clf.score(x_teste, y_teste)
res_arv_dec_clf = arv_dec_clf.score(x_teste, y_teste)

ensemble = VotingClassifier(estimators=[('bayes_clf', bayes_clf), ('knn_clf', knn_clf),('reg_log_clf', reg_log_clf), ('arv_dec_clf', arv_dec_clf)], voting='hard')
ensemble.fit(x_treino, y_treino)
ensemble = ensemble.score(x_teste, y_teste)

print('Bayes: ', res_bayes_clf*100)
print('KNN:',res_knn_clf*100)
print('Reg: ', res_reg_log_clf*100)
print('Arvore: ', res_arv_dec_clf*100)
print('Ensemble: ', ensemble*100)

