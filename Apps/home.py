import streamlit as st
import pandas as pd
import numpy as np


def app():
    st.title("H&M Personalized Fashion Recommendations")
    st.header('Provide product recommendations based on previous purchases')

    st.subheader('Context')
    st.markdown("""
    In the context of kaggle competition, H&M Group invites you to develop product recommendations based on data from previous transactions,
    as well as from customer and product meta data. The available meta data spans from simple data, such as garment type and customer age, 
    to text data from product descriptions, to image data from garment images.
    
    """)

    st.subheader('Purpose')
    
    st.markdown("""
    **Give a sequence of 12 purchases predictions for the 7 days after the training period of the model.**

    The metric used and imposed by H&M to rank competitors is MAP@12 ( Mean Average Precision).
    """)

    with st.expander("Principle AP@30"):
        st.image('.//data//figures_RNN//illu_MAP12.PNG',width=600)




