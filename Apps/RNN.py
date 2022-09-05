import streamlit as st
import base64
import pandas as pd

def app():
    
    st.title('Recommendation model with RNN (Recurrent neural network)')

    
    st.subheader('Strategy')
    st.markdown("""
    Make an analogy between a sequence of purchases by a customer and word predictions that form a coherent sentence.
    """)
    
    file_ = open('.//data//figures_RNN//contexte_fig.gif', "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="contexte gif" width="600">',
        unsafe_allow_html=True,
    )

    st.subheader('Dataset')
    df=pd.read_pickle('.//data//dataset_RNN.pkl')
    st.dataframe(df,height=500)

    st.subheader('Pipeline')

    st.image('.//data//figures_RNN//Pipeline.jpeg')


    with st.expander("sequences creation"):
        st.image('.//data//figures_RNN//creation_seq.png',width=600)

    with st.expander("Features - Targets"):
        st.image('.//data//figures_RNN//feature_target.png',width=600)
    
    with st.expander("Padding"):
        st.image('.//data//figures_RNN//padding.png',width=600)

    st.subheader('Architecture selected')

    st.image('.//data//figures_RNN//architecture_RNN.png')

    with st.expander("Model training"):
        st.image('.//data//figures_RNN//loss_accuracy.PNG',width=600)
