
from configuracoes import conf
import pandas as pd
from io import StringIO

def get_base_de_dados():
    '''
    Abre o arquivo da base de dados yeast1.dat, localizado na pasta base_de_dados e retorna o conteúdo do arquivo em um DataFrame.
    A função usa as constantes BASE_DE_DADOS e ROTULOS_ATRIBUTOS que estão localizados em configuracoes/conf.py
    '''
    #Transformando o arquivo dat em dataframe
    arquivo_txt = open(conf.BASE_DE_DADOS, 'r')

    base_de_dados = pd.DataFrame( 
        [[str(token) for token in line.split()]  
            for line in arquivo_txt if line.strip()],
        columns = conf.ROTULOS_ATRIBUTOS
    )
    '''
    Remove a primeira coluna do DataFrame
    '''
    base_de_dados = base_de_dados.drop(conf.ROTULOS_ATRIBUTOS[0], axis=1)
    
    return base_de_dados