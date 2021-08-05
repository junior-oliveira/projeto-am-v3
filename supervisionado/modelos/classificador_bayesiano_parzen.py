from os import X_OK
import numpy as np
import math
from configuracoes import conf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
import time
class ClassificadorBayesianoParzen(BaseEstimator, ClassifierMixin):
    
    def __init__(self, h=2, dim=8):
        self.h = h
        self.dim = dim
        self.kernel = self.hypercube
    
    def hypercube(self, k):
        """
        Hypercube kernel for Density Estimation.
        """
        return np.all(k < 0.5, axis=1)
        

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
        
        y = []
        
        for i, xk in enumerate(X):
            posteriores = []

            verossimilhancas = []

            for k in range(len(self.classes_)):
                
                wk = self.classes_[k]
                pxk_wr = self.pxk_wi(xk, wk, self.y_)
                verossimilhancas.append(pxk_wr)
                            
            posteriores = self.posteriores(verossimilhancas)
            indice = np.argmax(posteriores)
        
            y.append(self.classes_[indice])
        
        return y
    
    def get_probabilidades_a_priori(self, y):
        """ 
            Estima as probabilidades a posteriori a partir do vetor y 
        """
        
        classes = self.classes_
        probabilidades_a_priori = []
        for i in range(len(classes)):    
            probabilidades_a_priori.append (len(y[y == classes[i]])/len(y))
        probabilidades_a_priori = np.array(probabilidades_a_priori)
        
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
            self.mu_[k] = (self.mu_[k] / self.n_de_exemplos_classes_[k])
            
        #Estimação do vetor de variâncias
        for k in range(len(self.classes_)):
            for i in range(len(y)):
                if(self.classes_[k] == y[i]):   
                    self.variancias_[k] += np.linalg.norm(np.subtract(X[i],self.mu_[k]))**2    
            self.variancias_[k] = self.variancias_[k]/(self.n_de_exemplos_classes_[k]*self.n_atributos_)
        
  
        # Estimação das matrizes sigma e sigma_inv
        for k in range(len(self.classes_)):
            np.fill_diagonal(self.sigma_[k], self.variancias_[k])
            self.sigma_inv_[k] = np.linalg.inv(self.sigma_[k])
    
    def pxk_wi(self, xk, wi, y):
        """ 
        Parâmetros:
            xk - k-ézimo exemplo da base de treinamento para o qual se deseja estimar a densidade
            wi - classe na qual deve ser estimada a densidade de xk
            y - rótulos de todas as classes do conjunto de treinamento

        Retorno:
            Densidade estimada para xk através da janela de parzen
        """
        # Armazena todos os exemplos do conjunto de treinamento, pertencentes à classe wi
        exemplos_wi = self.X_[y == wi]
        
        # Retorna o valor estimado da densidade para o exemplo xk
        res = self.est_densidade_janela_de_parzen(exemplos_wi, xk, self.h, self.dim)
        
        return res

    def posteriores(self, verossimilhancas_x):
        """ 
        Parâmetros:
            verossimilhancas_x - array unidimensional contendo as verossimilhanças para todas as classes 
            dado um ponto x

        Retorno:
            Array contendo as probabilidades a posteriori estimada para todas as classes de um ponto x
        """
        posteriores = np.multiply(verossimilhancas_x, self.probabilidades_a_priori) / np.dot(verossimilhancas_x, self.probabilidades_a_priori)

        return posteriores
    
    def kernel_gaussiano_univariado(self, h, x, x_i, d):
        """
        Parâmetros:
            h: tamanho da janela
            x: ponto x para estimação de densidade (array numpy 'd x 1')
            x_i: exemplo do conjunto de treinamento (array numpy 'd x 1')
        
        Retorno:
            Array 'd x 1' com todos os valores da norma univariada  
        """
        
        # Verifica se os vetores possuem as mesmas dimensões
        assert (x.shape == x_i.shape), 'vetores x and x_i devem ter as mesmas dimensões'

        # Calcula a normal univariada para cada uma das d dimensões (d é o tamaho do vetor 'd x 1')
        normas_univariadas = (1/(np.sqrt(2*np.pi))) * np.exp(-(((x - x_i) / h)**2) /2)

        # Calcula o kernel multivariado do produto a partir do vetor de normas univariadas. 
        # Uma vez que as normas já foram calculadas, basta calcular o produtório
        kernel_multiv_prod = np.prod(normas_univariadas)
        return kernel_multiv_prod


    def est_densidade_janela_de_parzen(self, exemplos, ponto_x, h, d):
        """
        Estimador Janela de Parzen

        Parâmetros:
            exemplos: array numpy 'n x d'-dimensional, cada amostra armazenada em uma linha separada
            ponto_x: ponto x para estimação de densidade, array numpy 'd x 1'-dimensional
            h: largura da janela
            d: dimensões (quantidade de atributos)

        Retorno:
            Desidade estimada p(x) referente ao ponto x

        """
        #k_n representa o número de exemplos dentro da região
        k_n = 0
        for row in exemplos:
            k_n += self.kernel_gaussiano_univariado(h=h, x=ponto_x, x_i=row, d=d) 
        return (k_n / len(exemplos)) / (h**d)