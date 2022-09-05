import streamlit as st
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from dtreeviz.trees import dtreeviz
from sklearn.base import TransformerMixin
import streamlit.components.v1 as components

def app():
    
    
    ### Resources
    search_results = pd.read_pickle('resources/gnn_hp_search.pkl')
    
    st.title('Graph Neural Network')

    st.markdown("""
We have seen in the previous sections that the purchase history of customers, as well as certain sequential aspects such as the purchase of one item after another, had something to do with the prediction to accomplish.
We therefore sought to train a model that was able to deal with these relationships: We therefore decided to look into Graph Neural Networks.
    """)
    
    st.header('Principle')
        
    st.markdown("""
The principle of the GNN that we implement here consists of three stages: the creation of the graph, the calculation of an embedding for each node of the graph, and finally a link prediction according to the embeddings created.

The structure of the graph that we use here is a bipartite heterogeneous article – customers structure, between which the links symbolize the purchases.
    """)
    
    
    st.image('resources/gnn.png')
    
    st.subheader('Message passing')
    
    st.markdown("""
Nodes and links can have their own features. On this basis, an embedding is made for each node according to the principle of message passing: the information of the neighboring nodes is aggregated according to a mathematical formula involving a certain number of parameters (below), forming what we call a *message*. This message is then agregated with the node informations, this forming a new embedding. The operation can be repeated several times, the present model being able to involve between 2 and 5 layers of aggregation.
    """)
    
    col_1, col_2 = st.columns(2) 
    col_1.latex('\\text{Edge-wise: } m_{e}^{(t+1)} = \phi \left( x_v^{(t)}, x_u^{(t)}, w_{e}^{(t)} \\right) , ({u}, {v},{e}) \in \mathcal{E}.')
    col_1.latex('\\text{Node-wise: } x_v^{(t+1)} = \psi \left(x_v^{(t)}, \\rho\left(\left\lbrace m_{e}^{(t+1)} : ({u}, {v},{e}) \in \mathcal{E} \\right\\rbrace \\right) \\right).')
    col_1.image('resources/Message passing.png')
   
    st.markdown("""We use the following hyperparameters for message calculation:
             
| Name      | Description |   
| ----------- | ----------- |     
|mean | Mean aggregation of all neightbors / link informations followed by Dense layer.|
|mean_nn | Same preceded by a preaggregation Dense layer.|
|mean_nn_w | Same with a multiplication correponsding to the weight of the link (= number of repurchases)|
|pool_nn | Pool aggregation of all neightbors informations, preceded by a preaggregation Dense layer, and followed by a Dense layer.|
|mean_nn_w | Same with a multiplication correponsding to the weight of the link (= number of repurchases)|
  

                """)

    
    st.markdown("""
&nbsp;                  
The Neightbor -> target node aggregation could be done either with sum, mean or max.""")
  
    st.header('Tuning')
    
    st.markdown("""
A hyperparameter search was performed using a Bayesian approach. It shows that a 2-layer model with poll_nn / pool_nn_w aggregation function and a small embedding dimension might be the best for this prediction task.
""")
    
    expander = st.expander('See search results as a DataFrame')
    expander.dataframe(search_results)
    
    st.pyplot(hp_search(search_results))
    
    st.header('Results')
    
    st.markdown("""
Although the average of the model's precision at 3, 6, and 12 exceeded 3% on hyperparameters search, the final training result barely exceeded 2% precision at 12.
More importantly, the model could not achieve good precision on the top-ranked articles. This is all the more penalizing since they represent half of the score when calculating the MAP@12, the first even counting for 1/4 of the points.
    """)
    
    st.image('resources/gnn_tree.png')
    

#@st.cache(suppress_st_warning=True)
def hp_search(search_results) -> plt.figure:
    

    non_numeric = ['Aggregation of neighbors', 'Aggregation node / neighbors', 'aggregator_weighted', 'Dimensions', 'Embedding layer', 'Reduce article features', 
                'Number of layers',
        'Normalization', 'out_dim', 'hidden_dim', 'Precision', 'lost_patience']
    
    fig, axes = plt.subplots(5, 2, figsize=(18, 31))

    for i in range(len(search_results.columns) - 1):
        column = search_results.columns[i]
        if column in non_numeric:
                sns.boxplot(ax = axes[i // 2][i % 2], x = search_results.columns[i], y = 'Precision',  data = search_results)
        else :
                sns.scatterplot(ax = axes[i // 2][i % 2], x = search_results.columns[i], y = 'Precision', alpha = 0.5, data = search_results)
    
    return fig