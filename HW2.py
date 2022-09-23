import streamlit as st
import seaborn as sns
import plotly.express as px


df_iris = sns.load_dataset("iris")
fig = px.scatter_3d(df_iris, x="sepal_length", y="sepal_width", 
                    z="petal_width", color="species")
st.plotly_chart(fig, use_container_width=True)