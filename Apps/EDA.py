import streamlit as st
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def app():
    
    
    ### Load resources
    
    articles = pd.read_pickle('resources/articles_sampled.pkl')
    customers = pd.read_pickle('resources/customers_sampled.pkl')
    transactions = pd.read_pickle('resources/transactions_sampled.pkl')
    week_board = pd.read_pickle('resources/week_board.pkl')
    customer_sales_board = pd.read_pickle('resources/customer_sales_board.pkl')
    last_week_sales = pd.read_pickle('resources/last_week_sales.pkl')
    
    customer_base_columns = ['customer_id', 'FN', 'Active', 'club_member_status', 'fashion_news_frequency', 'age', 'postal_code']
    customer_sales_columns = ['customer_id', 'average_cart_articles', 'average_cart_price', 'total_carts', 'total_articles', 'total_price', 'average_cart_interval', 'repurchases', 'repurchases_interval']
    customer_prefs_columns = ['baby', 'divided', 'ladieswear', 'menswear', 'sport', 'age_around_15', 'age_around_25', 'age_around_35', 'age_around_45', 'age_around_55', 'age_around_65', 'postal_code_group', 'group']
    
    article_base_columns = ['article_id', 'product_code', 'prod_name', 'product_type_name', 'product_group_name', 'graphical_appearance_name', 'colour_group_name', 'perceived_colour_value_name', 'perceived_colour_master_name', 'department_name', 'index_code', 'index_name', 'index_group_name', 'section_name', 'garment_group_name', 'detail_desc']
    article_sales_columns = ['total_purchases', 'average_quantity', 'repurchases', 'repurchase_interval','average_price', 'age_around_15', 'age_around_25', 'age_around_35', 'age_around_45', 'age_around_55', 'age_around_65', ]
    
    transaction_base_columns = ['t_dat', 'customer_id', 'article_id', 'price', 'sales_channel_id']
    
    p_values = pd.read_pickle("resources/p_values.pkl")
    
    
    st.title('Exploratory Data Analysis')
    
    
    ### Key Figures ###
    st.header('Key informations')
    
    metric_1, metric_2, metric_3 = st.columns(3)
    
    metric_1.metric("Customers", '1.35M')
    metric_2.metric("Articles", "104 000")
    metric_3.metric("Transactions", "27M")

    
    st.markdown("""**Transactions**  
The transaction table contains 2 years of customers by-article purchase history : date, customer and article IDs, price and sales channel.
 """)        
    expander = st.expander("See transactions DataFrame")   
    expander.dataframe(transactions[transaction_base_columns])
    st.markdown("""**Customers**  
The customers table contains few informations : age, subscription to fashion newsletter (with frequency), member status, and postal code in an encoded format."
     """)                
    expander = st.expander("See Customers DataFrame")         
    expander.dataframe(customers[customer_base_columns])
    st.markdown("""**Articles**  
Articles table contains mainly categorical features, like product / garment type, colour, section etc, along with product name and description in text format.
    """)
    expander = st.expander("See Articles DataFrame")   
    expander.dataframe(articles[article_base_columns])
    

    ### What data to use ? ###
    st.header('What data to use ?')
    
    st.markdown("""The challenge consists in predicting what articles the customers will purchase **on the first week following the training set**.
                Starting from this, we need to figure out what data are likely to be used to train our model with.""")
    
    week_sales_heatmap = heatmap(week_board.corr(), center  = 0.8)
    customer_sales_heatmap = heatmap(customer_sales_board.corr(), center  = 0.9)
    
    st.subheader('Weekly sales correlations')
    
    st.markdown("""
                   One first idea is to compare the sales by articles for different weeks, and check what correlation we obtain. 
                   Thus, the following graphs shows that the weeks 0 & 1 (= the two lasts of the training set) might be similar to the prediction week. 
                   On the other hand, training the model on the same week one year before (week 51) seems not to be a good idea.
                   """)
    
    expander = st.expander("See DataFrame")
    expander.dataframe(week_board)
    col_1, col_2 = st.columns(2)
    col_1.pyplot(week_sales_heatmap)
    
    st.subheader('Customers sales correlations')
    st.markdown("""
                We analyzed the correlation between sales by articles for different subset of customers, each group on the graph below representing 1% of them.
                It shows that it might not be bad to train / validate the model on a fraction of customers.
                """)
        
    expander = st.expander("See DataFrame")
    expander.dataframe(customer_sales_board)
    col_1, col_2 = st.columns(2)
    col_1.pyplot(customer_sales_heatmap)
    

    st.subheader('Article cumulative sales')
    st.markdown("""
                    50% of the sales are covered by 1000 articles or 1% of all references, whereas 7000 references covers 90% of the total sales.
                    Taking this into account, it seems possible to train a model on a well-chosen small part of the articles set.
""")
    col_1, col_2 = st.columns(2)
    col_1.pyplot(article_cumulative_sales(last_week_sales))
    
    
    st.subheader('P values between features')
    st.markdown("""
                    A systematic p-values calculation reveals that the differents fields are weakly but mostly interconnected.
    """)
    col_1, col_2 = st.columns(2)
    col_1.pyplot(heatmap(p_values, cbar = False))



    ### Feature extraction ###
    st.header('Feature extraction')
    st.subheader('Sales history aggregation')
    
    st.markdown("""
    Additional features have been added to customers and articles tables, mainly by aggregating transactions informations.
    
    - **For customers :** total purchases / carts / spended amount,  average cart amount, average article number by cart, mean interval between two purchases, categorical preferences (lady / men / sport etc).
""")
    
    st.dataframe(customers[customer_sales_columns + customer_prefs_columns].head(5))
    st.markdown("""
    - **For articles :** total puchases / amount, average price, ratio by customer's age.
    
    Some more systematic feature extraction could have been performed.
    """)
    st.dataframe(articles[article_sales_columns].head(5))

    st.subheader('Customers clustering')
    st.markdown("We added some clustering (KMean) in order to define customers groups. This feature appeared to be useful for establishing an articles shortist.")
    col_1, col_2 = st.columns(2)
    col_1.pyplot(customer_clustering())


# Fonction de génération de heatmap
# @st.cache(suppress_st_warning=True)
def heatmap(dataframe: pd.DataFrame, center = 0, vmax = 1, cbar = True) -> plt.figure:
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(dataframe, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(6, 5))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(dataframe, cmap=cmap, vmax=vmax, center=center,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax = ax, cbar = cbar)
    
    return f
    
    
# Affichage d'un graphe avec part cumulative des articles sur les ventes totales
# @st.cache(suppress_st_warning=True)
def article_cumulative_sales(last_week_sales: pd.DataFrame()) -> plt.figure:

    total_sales = last_week_sales['sales'].sum()

    cumulative_sales = [0]
    index = 0

    while index < 10000:
        sales = last_week_sales.iloc[index, 1]
        previous_sum = cumulative_sales[-1]
        cumulative_sales.append(sales + previous_sum)
        
        index += 1
        
    x = range(0, len(cumulative_sales))

    f = plt.figure(figsize = (7, 4))
    plt.plot(x, cumulative_sales / total_sales * 100)
    plt.xlabel("Article number")
    plt.xlim((0, 10000))
    plt.ylabel("Cumulative sales (%)")
    plt.ylim((0, 100))

    plt.grid(True)
    
    return f
    
    
#@st.cache(suppress_st_warning=True)   
def customer_clustering():
    score_curve = pd.read_pickle('../streamlit/resources/customer_clustering.pkl')
    
    f = plt.figure(figsize=(8, 4))
    plt.plot(range(1, 15), score_curve, "bo-")
    plt.xlabel("Number of clusters", fontsize=14)
    plt.ylabel("MAP@12 score", fontsize=14)
    
    return f