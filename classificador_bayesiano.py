import numpy as np
import math
from configuracoes import conf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
class ClassificadorBayesiano(BaseEstimator, ClassifierMixin):

    def fit(self, X, y):
        
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Obtém o número de classes
        self.n_classes_ = len(self.classes_)

        # Obtém o número de atributos
        self.n_atributos_ = X.shape[1]
        
        # Obtém o número de exemplos da amostra
        self.numero_de_exemplos_ = y.shape[0]

        # Obtém o número de exemplos em cada classe
        (unique, self.n_de_exemplos_classes_) = np.unique(y, return_counts=True)

        # Inicializa os arrays de parâmetros
        self.mu_  = np.zeros((self.n_classes_, self.n_atributos_))
        self.variancias_ = np.zeros(self.n_classes_)
        self.sigma_ = np.zeros((self.n_classes_,self.n_atributos_,self.n_atributos_))
        self.sigma_inv_ = np.zeros((self.n_classes_,self.n_atributos_,self.n_atributos_))

        self.X_ = X
        self.y_ = y
        
        # Obtém as probabilidades a priori
        self.probabilidades_a_priori = self.get_probabilidades_a_priori(y)
        
        self.estimacao_parametros(X, y)
        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        n_atributos_ = X.shape[0]
        y = np.empty(n_atributos_, dtype=object)
        d = self.n_atributos_
        for i, xk in enumerate(X):
            max_pwi_xk = 0.0
            classe_predita = ''
            for k in range(len(self.classes_)):
                pwi = self.probabilidades_a_priori[k]
                
                mu_i = self.mu_[k]
                sigma_inv_i = self.sigma_inv_[k]
                
                pwi_xk = self.pwi_xk(xk, pwi, mu_i, sigma_inv_i, d)
                if(pwi_xk > max_pwi_xk):
                    max_pwi_xk = pwi_xk
                    classe_predita = self.classes_[k]
            y[i] =  classe_predita
        c, i = np.unique(y, return_inverse=True)
        #closest = np.argmin(euclidean_distances(X, self.X_), axis=1)  

        #substituir por return y em caso de problemas. 
        return self.classes_[i]
    
    def get_probabilidades_a_priori(self, y):
        """ 
            Estima as probabilidades a posteriori a partir do vetor y 
        """
        classes = self.classes_
        probabilidades_a_priori = []
        for i in range(len(classes)):    
            probabilidades_a_priori.append (len(y[y == classes[i]])/len(y))    
        return probabilidades_a_priori

    def estimacao_parametros(self, X, y):
        """ 
            Estima os vetores de médias e a matrizes de variância e covariância
        """
        check_is_fitted(self)
        # Armazena os vetores de médias e a matrizes de variância e covariância
        

        #Estimação do vetor de médias
        
        for k in range(len(self.classes_)):  
            for i in range(len(y)):
                if(self.classes_[k] == y[i]):                 
                    self.mu_[k] = np.add(self.mu_[k], X[i])
        for k in range(len(self.classes_)):
            self.mu_[k] = self.mu_[k] / self.n_de_exemplos_classes_[k]           

        #Estimação do vetor de variâncias
        for xk in X:
            for k in range(len(self.classes_)):
                self.variancias_[k] += np.linalg.norm(np.subtract(xk,self.mu_[k]))**2    
        self.variancias_ = self.variancias_/(self.numero_de_exemplos_*self.n_atributos_)
        
        # Estimação das matrizes sigma e sigma_inv
        for k in range(len(self.classes_)):
            np.fill_diagonal(self.sigma_[k], self.variancias_[k])
            self.sigma_inv_[k] = np.linalg.inv(self.sigma_[k])
        
    # Calcula a norma conforme especificado no trablho. Verificar se o resultado melhora em relação a np.linalg.norm
    def norma(self, vetor):    
        soma = 0.0    
        for elem in vetor:
            soma += elem
        return soma
    
    def pxk_wi(self, xk, mu_i, sigma_inv_i, d):     
        """ 
        xk - k-ésimo exemplo do conjunto de aprendizagem 
        mu_i - vetor de médias da classe wi
        sigma_inv_i - matriz de variância e covariância inversa da classe wi
        d - número de atributos do conjunto de aprendizagem
        """
        res = ((2 * math.pi) ** (-d/2)) * (np.linalg.det(sigma_inv_i) ** (0.5)) * math.exp(-0.5 * np.dot(np.dot((xk - mu_i), sigma_inv_i), (xk - mu_i).T))
        
        return res

    def pwi_xk(self, xk, pwi, mu_i, sigma_inv_i, d):
        """ 
        xk - k-ésimo exemplo do conjunto de aprendizagem 
        mu_i - vetor de médias da classe wi
        sigma_inv_i - matriz de variância e covariância inversa da classe wi
        pwi - probabilidade a priori da classe wi
        d - número de atributos do conjunto de aprendizagem
        """
        evidencia = 0.0
        for r in range(len(self.classes_)):
            pwr = self.probabilidades_a_priori[r]
            pxk_wr = self.pxk_wi(xk, self.mu_[r], self.sigma_inv_[r], d)
            evidencia += pwr * pxk_wr

        res = (self.pxk_wi(xk, mu_i, sigma_inv_i, d) * pwi)/evidencia

        return res
    
