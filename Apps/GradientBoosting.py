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
    
    ### Resources ###
    
    
    search_results = pd.read_pickle('resources/lightgbm_hp_search.pkl')
    
    # ['colsample_bytree', 'Feature set', 'min_child_samples',
    #   'min_child_weight', 'num_leaves', 'reg_alpha', 'reg_lambda',
    #   'subsample', 'objective', 'metric', 'eval_at', 'map@12', 'Best score']
    
    search_results = search_results[['map@12', 'Feature set', 'objective', 'reg_alpha', 'reg_lambda', 'num_leaves','min_child_weight', 'min_child_samples', 'colsample_bytree','subsample']]\
        .dropna()\
        .sort_values('map@12', ascending = False)
            
    
    
    data_train = pd.read_pickle('resources/lightgbm_train.pkl')
    
    
    
    st.title('Gradient boosting')
    
    
    st.header('Principle')
    
    st.markdown("""
The first modeling that we put in place consists of identifying a restricted selection of articles that is likely to cover the majority of customer purchases, and training a Gradient Boosting algorithm  to recognize the items that have actually been purchased by users.

**It turned out that this approach is that of the winners of the Kaggle challenge.** The first-place team indeed trained several LightGBM classifiers, varying the size of the article selection as well as the number of weeks used for training. (we also thank the teams at the top of the ranking for having all given their solution after the competition!)
For our part, we have chosen to train a single classifier from a selection of 100 articles per customer (= dynamic selection), over the last week of the dataset, using 20% of customers for validation.       
    """)
    
    st.image('resources/lightgbm.png')
    
    ### Shortlist
    st.header('Selecting top 100 articles')
    
    st.markdown(""" 
We therefore sought a restricted selection of 100 articles per customer on which to train our model.
For this we have aggregated different intermediate lists according to their potential, which we have measured through the maximum precision at 12 that they can obtain: from 0 if none of the items in the list have been purchased by the customer, to 1 if the list contains either 12 or all of the customer's purchases.               
""")
    col_1, col_2 = st.columns(2)
    col_1.pyplot(article_curves())
    
    st.subheader("Here's how the final dataset looks like :")
    st.markdown(""" 
                Each sample corresponds to a customer-article pair, crossing the 70,000 customers who made a purchase in the last week with the 100 selected items.
The task assigned to the model is thus to predict whether the customer bought the item or not during the last week of the training set.

The ~1500 customers with more than 500 items, 100 baskets or 50 repurchases were considered outliers and removed from the training set.

Negative/Positive sample ratio has been lowered to 50 due to hardware limitation. The dataset thus contains 1,800,000 samples, including 36,000 positives.
    """)
    st.dataframe(data_train)
    
    ### Hyperparameter Search
    st.header('Tuning')
    st.markdown("""
A hyperparameter search was performed using a Bayesian approach, using the main lightGBM parameters along with 5 different set of features.
""")
    expander = st.expander("See search results as a Dataframe")
    expander.dataframe(search_results)
    st.pyplot(hp_search(search_results))
    
    
    ### Interpretation
    st.header('Results')
    st.markdown("""
**The model obtained a MAP@12 score of 0.01837**.  

A decision-tree-based interpretation on validation set shows that most of the positive example are recognized by our model.  
On the other hand, not all of them are in top position which is quite penalizing according to the MAP@12 score calculation.  
""")
    st.image('resources/lightgbm_tree.png')
    
    

def hp_search(search_results) -> plt.figure:
    
    fig, axes = plt.subplots(5, 2, figsize=(24, 36))
    
    categories = ["Feature set", "objective"]
    log_columns = ["reg_lambda", "min_child_weight", 'reg_alpha']

    for i in range(len(search_results.columns) - 1):
        column = search_results.columns[i + 1]
       
        if column in log_columns:
            axes[i // 2][i % 2].set(xscale="log")
       
        if column in categories: 
            sns.boxplot(ax = axes[i // 2][i % 2], x = search_results.columns[i + 1], y = 'map@12', data = search_results)
        else: 
            sns.scatterplot(ax = axes[i // 2][i % 2], x = search_results.columns[i + 1], y = 'map@12', alpha = 0.5, data = search_results)
            
    return fig
    
#@st.cache(suppress_st_warning=True)
def article_curves() -> plt.figure:
    lists = pd.read_pickle('resources/article_curves.pkl')
    
    x = np.arange(0, 101, 5)

    f = plt.figure()
    plt.plot(x, lists['Global top sales'], label = "Global top sales")
    plt.plot(x, lists['Final list'], label = "Final list")
    plt.plot(x, lists['Already purchased'], label = "Already purchased")
    plt.plot(x, lists['Cross'], label = "Age + Category + Cluster")

    plt.xlabel("Number of articles")
    plt.ylabel("Max precision@12")
    plt.legend()

    return f


#@st.cache(suppress_st_warning=True)
def treeviz(top_rank_threshold: int):
    data_to_analyze = pd.read_pickle('resources/lightgbm_prediction_data.pkl')
    
    feature_set = ['in_pair_list',
       'in_repurchase_list', 'in_cross_list', 
       'product_group_name',
       'perceived_colour_value_name',
       'perceived_colour_master_name', 
       'index_name', 'index_group_name', 'section_name', 'garment_group_name',
       'total_purchases', 'average_quantity', 'average_price',
       'has_image', 'age_around_15_article', 'age_around_25_article',
       'age_around_35_article', 'age_around_45_article',
       'age_around_55_article', 'age_around_65_article', 'repurchases_article',
       'repurchase_interval', 'FN', 'Active', 'club_member_status',
       'fashion_news_frequency', 'age', 'average_cart_articles',
       'average_cart_price', 'total_carts', 'total_articles', 'total_price',
       'average_cart_interval', 'baby', 'divided', 'ladieswear', 'menswear',
       'sport', 'repurchases_customer', 'repurchases_interval',
       'age_around_15_customer', 'age_around_25_customer',
       'age_around_35_customer', 'age_around_45_customer',
       'age_around_55_customer', 'age_around_65_customer', 'postal_code_group',
       'group', 'age_ratio', 'index_ratio']
    
    data_to_analyze['rank_class'] = data_to_analyze['class']
    data_to_analyze.loc[
        (data_to_analyze['class'] == 2) & 
        (data_to_analyze['position_in_prediction'] < top_rank_threshold),
        'rank_class'
    ] = 3 # Faux positifs en top position
    data_to_analyze.loc[
        (data_to_analyze['class'] == 4) & 
        (data_to_analyze['position_in_prediction'] < top_rank_threshold),
        'rank_class'
    ] = 5 # Vrais positifs en top position

    # One-hot encoding des catégories pour pouvoir utiliser les données dans un DecisionTree


    dummify = Dummify()

    features = dummify.transform(data_to_analyze[feature_set])
    

    explainer = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2000)
    explainer.fit(features, data_to_analyze['rank_class'])
    

    class_colors = [None, # 0 classes
                    None, # 1 class
                    ["#FEFEBB","#a1dab4"], # 2 classes
                    ["#ff0000","#00ff00",'#0000ff'], # colors were changed here
                    ["#FEFEBB","#D9E6F5",'#a1dab4','#fee090'], # 4
                    ["#FEFEBB", "#e2ecf7", '#b5cde8','#bce4c9','#80ca99'], # 5
                    ["#FEFEBB",'#c7e9b4','#41b6c4','#2c7fb8','#fee090','#f46d43'], # 6
                    ["#FEFEBB",'#c7e9b4','#7fcdbb','#41b6c4','#225ea8','#fdae61','#f46d43'], # 7
                    ["#FEFEBB",'#edf8b1','#c7e9b4','#7fcdbb','#1d91c0','#225ea8','#fdae61','#f46d43'], # 8                                       ["#FEFEBB",'#c7e9b4','#41b6c4','#74add1','#4575b4','#313695','#fee090','#fdae61','#f46d43'], # 9
                    ["#FEFEBB",'#c7e9b4','#41b6c4','#74add1','#4575b4','#313695','#fee090','#fdae61','#f46d43','#d73027'] # 10
    ]
    viz = dtreeviz(explainer, features, data_to_analyze['rank_class'],
                    target_name="Prediction status",
                    feature_names=features.columns,
                    colors = {'classes': class_colors},
                    class_names=['False negative', 'False positive', 'False positive in top position', 'True positive', 'True positive in top position']
                )
    
    
    return viz



class Dummify(TransformerMixin):
    def __init__(self):
        return

    def fit(self):
        return self

    def transform(self, dataset):

        for column in dataset.columns:
            if not isinstance(dataset[column].dtype, pd.CategoricalDtype):
                continue

            dummies = pd.get_dummies(
                dataset[column], prefix=column, prefix_sep=":")
            dataset = pd.concat([dataset, dummies], axis=1)
            dataset.drop(columns=column, axis=1, inplace=True)

        return dataset

    def inverse_transform(self, dataset):

        columns = dataset.columns

        for column in columns:
            if ':' not in column:
                continue

            category_name = column.split(':')[0]
            label = column.split(':')[1]

            if category_name not in dataset.columns:
                dataset[category_name] = ''

                dataset[category_name] = dataset[[category_name, column]].apply(
                    lambda x: label if x[column] == 1 else x[category_name])

                dataset.drop(column=column, axis=1, inplace=True)


#@st.cache(suppress_st_warning=True)
def st_dtree(plot, height=None):

    dtree_html = f"<body>{plot.svg()}</body>"

    components.html(dtree_html, height=height)