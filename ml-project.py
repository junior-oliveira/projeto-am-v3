#import time
from utils import utils
from sklearn.model_selection import train_test_split
from classificador_bayesiano import ClassificadorBayesiano
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from modelos.classificador_bayesiano_parzen import ClassificadorBayesianoParzen


dados = utils.get_base_de_dados()
rotulos_yeast = dados['SEQUENCE NAME']
atributos_yeast = dados.drop('SEQUENCE NAME', axis=1)

# Normalização dos dados
atributos_yeast_norm = utils.normalizar_dados(atributos_yeast)
atributos_yeast = atributos_yeast_norm 

x_treino, x_teste , y_treino, y_teste = train_test_split(atributos_yeast,rotulos_yeast,test_size=0.3, stratify=rotulos_yeast)
NB = ClassificadorBayesiano()
NB.fit(x_treino, y_treino)

# Implementação do classificador bayesiano de acordo com o que foi pedido na disciplina de AM
bayes_clf = ClassificadorBayesiano()
knn_clf = KNeighborsClassifier(n_neighbors=5)
reg_log_clf = LogisticRegression(max_iter=200)
bayes_parzen_clf = ClassificadorBayesianoParzen(h=0.1)


bayes_clf.fit(x_treino, y_treino)
knn_clf.fit(x_treino, y_treino)
reg_log_clf.fit(x_treino, y_treino)
bayes_parzen_clf.fit(x_treino, y_treino)


res_bayes_clf = bayes_clf.score(x_teste, y_teste)
res_knn_clf = knn_clf.score(x_teste, y_teste)
res_reg_log_clf = reg_log_clf.score(x_teste, y_teste)
res_bayes_parzen_clf = bayes_parzen_clf.score(x_teste, y_teste)

ensemble = VotingClassifier(estimators=[('bayes_clf', bayes_clf), ('knn_clf', knn_clf),('reg_log_clf', reg_log_clf), ('bayes_parzen', bayes_parzen_clf)], voting='hard')
ensemble.fit(x_treino, y_treino)
ensemble = ensemble.score(x_teste, y_teste)

print('Bayes: ', res_bayes_clf*100)
print('KNN:',res_knn_clf*100)
print('Reg: ', res_reg_log_clf*100)
print('Bayes com Janela de Parzen: ', res_bayes_parzen_clf*100)
print('Ensemble: ', ensemble*100)
 
 
bayes_clf.fit(atributos_yeast, rotulos_yeast)