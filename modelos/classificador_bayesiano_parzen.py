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
        
        #for xk in X:
        #    for k in range(len(self.classes_)):
        #        self.variancias_[k] += np.linalg.norm(np.subtract(xk,self.mu_[k]))**2    
        #self.variancias_ = self.variancias_/(self.numero_de_exemplos_*self.n_atributos_)
        
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
    
    def pxk_wi(self, xk, wi, y):
        exemplos_wi = self.X_[y == wi]
        #t = time.time()
        res = self.parzen_estimation(exemplos_wi, xk, self.h, self.dim, self.parzen_window_func, self.gaussian_kernel)
        #elapsed = time.time() - 1
        #print('tempo gasto: ', elapsed) 
        return res

    def posteriores(self, verossimilhancas):
        """ 
        verossimilhancas - array unidimensional contendo as verossimilhanças para todas as classes
        """
        posteriores = np.multiply(verossimilhancas, self.probabilidades_a_priori) / np.dot(verossimilhancas, self.probabilidades_a_priori)

        return posteriores
    
    def hypercube_kernel(self, h, x, x_i):

        """
        Implementation of a hypercube kernel for Parzen-window estimation.

        Keyword arguments:
            h: window width
            x: point x for density estimation, 'd x 1'-dimensional numpy array
            x_i: point from training sample, 'd x 1'-dimensional numpy array

        Returns a 'd x 1'-dimensional numpy array as input for a window function.

        """
        assert (x.shape == x_i.shape), 'vectors x and x_i must have the same dimensions'
        return (x - x_i) / (h)
    
    def gaussian_kernel(self, h, x, x_i, d):
       
        """
        Implementation of a hypercube kernel for Parzen-window estimation.

        Keyword arguments:
            h: window width
            x: point x for density estimation, 'd x 1'-dimensional numpy array
            x_i: point from training sample, 'd x 1'-dimensional numpy array

        Returns a 'd x 1'-dimensional numpy array as input for a window function.

        """
        assert (x.shape == x_i.shape), 'vectors x and x_i must have the same dimensions'

        res = (1/(math.sqrt(2*math.pi)**d)*(h**d)) * np.exp(-0.5 * (((x - x_i) / h)**2))
        
        return res
        


    def parzen_window_func(self, x_vec, h=1):
        """
        Implementation of the window function. Returns 1 if 'd x 1'-sample vector
        lies within inside the window, 0 otherwise.

        """
        for row in x_vec:
            if np.abs(row) > (1/2):
                return 0
        return 1


    def parzen_estimation(self, x_samples, point_x, h, d, window_func, kernel_func):
        """
        Implementation of a parzen-window estimation.

        Keyword arguments:
            x_samples: A 'n x d'-dimensional numpy array, where each sample
                is stored in a separate row. (= training sample)
            point_x: point x for density estimation, 'd x 1'-dimensional numpy array
            h: window width
            d: dimensions
            window_func: a Parzen window function (phi)
            kernel_function: A hypercube or Gaussian kernel functions

        Returns the density estimate p(x).

        """
        #k_n representa o número de exemplos dentro da região
        k_n = 0
        for row in x_samples:
            x_i = kernel_func(h=h, x=point_x, x_i=row, d=d)
            k_n += window_func(x_i, h=h)
        return (k_n / len(x_samples)) / (h**d)