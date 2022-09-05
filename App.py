import streamlit as st
from multiApp import MultiApp
from Apps import home, ALS, GNN, RNN, EDA, GradientBoosting
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide")

#count = st_autorefresh(interval=1000, limit=1000, key="fizzbuzzcounter")


app = MultiApp()

app.add_app("Home", home.app)
app.add_app("Data Analysis", EDA.app)
app.add_app("Gradient Boosting", GradientBoosting.app)
app.add_app("ALS", ALS.app)
app.add_app("RNN", RNN.app)
app.add_app("Graph Neural Network", GNN.app)


app.run()