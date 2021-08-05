import matplotlib.pyplot as plt
import seaborn as sns
from configuracoes import conf
import umap
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def correlacao(df):
    """Plota a matriz de correlação para cada par de colunas no dataframe.

    Input:
        df: pandas DataFrame
    """

    plt.figure(figsize=(8,4)) 
    heatmap = sns.heatmap(df.corr(), annot=True, cmap="BuPu")
    heatmap.set_title('Mapa de correlação', fontdict={'fontsize':12}, pad=12)
    
def grafico_pareado(atributos_yeast, rotulos_yeast):
    """Plota dos atributos em relação aos rótulos das classes.

    Input:
        atributos_yeast: DataFrame de atributos 
        rotulos_yeast: DataFrame de Rótulos
    """
    nome_classe = conf.ROTULOS_ATRIBUTOS[9]
    dados_yeast_norm = atributos_yeast.copy()
    dados_yeast_norm.insert(8, nome_classe, rotulos_yeast.copy())    
    sns.pairplot(dados_yeast_norm, hue = nome_classe)

def grafico_2d_umap(atributos_yeast, rotulos_yeast):  
    rotulos_yeast = rotulos_yeast.map({"CYT":1, "NUC":2, "MIT":3, "ME3":4, "ME2":5, "ME1":6, "EXC":7, "VAC":8, "POX":9, "ERL":10})
    reducer = umap.UMAP(random_state=42)
    dados_reducao = atributos_yeast
    reducer.fit(dados_reducao)

    umap.UMAP(a=None, angular_rp_forest=False, b=None,
     force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
     local_connectivity=1.0, low_memory=False, metric='euclidean',
     metric_kwds=None, min_dist=0.1, n_components=2, n_epochs=None,
     n_neighbors=15, negative_sample_rate=5, output_metric='euclidean',
     output_metric_kwds=None, random_state=42, repulsion_strength=1.0,
     set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
     target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
     transform_queue_size=4.0, transform_seed=42, unique=False, verbose=False)

    embedding = reducer.transform(dados_reducao)
    # Verify that the result of calling transform is
    # idenitical to accessing the embedding_ attribute
    assert(np.all(embedding == reducer.embedding_))
    embedding.shape

    plt.scatter(embedding[:, 0], embedding[:, 1], c=rotulos_yeast, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(1,12)-0.5).set_ticks(np.arange(1,11))
    #plt.title('UMAP Projeção 2D da base Glass', fontsize=24);

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i]+5,y[i], ha = 'center')

def pandas_bar_plot(y, color_opt='grayscale'):
    cores = ['#000000', '#444444', '#555555', '#666666', '#777777', '#888888', '#999999', '#aaaaaa', '#bbbbbb', '#cccccc']
    if(color_opt == 'grayscale'):
        cores = ['#000000', '#444444', '#555555', '#666666', '#777777', '#888888', '#999999', '#aaaaaa', '#bbbbbb', '#cccccc']
    
    frequencias_classes = np.array([(y == idx).sum() for idx in conf.ROTULOS_CLASSES])
    height = frequencias_classes
    bars = conf.ROTULOS_CLASSES
    y_pos = np.arange(len(bars))
    fig, ax = plt.subplots()
    plt.title('Quantidade de instâncias por classe')
    plt.xlabel('Classes')
    plt.ylabel('Quantidade')
    plt.grid(False)
    # Create bars
    
    plt.bar(y_pos, height, width=.5)
    addlabels(y_pos, height)

    # Create names on the x-axis
    plt.xticks(y_pos, bars)

    # Show graphic
    plt.show()

