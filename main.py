import numpy as np
import pandas as pd
import streamlit as st

from categorical_fetures_dict import features_dic
from graph_functions import pie, scatter



static_df = pd.read_csv("dataset.csv",sep=",")

df = static_df.copy()
df.set_index("CLIENTNUM",inplace=True,drop=True)
df = df.iloc[:,:-4].drop(["Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1"],axis=1)

df.replace(features_dic,inplace=True)
df["Income_Category"].replace(0,df["Income_Category"].mean(),inplace=True)






st.header('Clients Classification')
st.write("Who will close the account?")
st.write('Dataset source: www.kaggle.com/sakshigoyal7/credit-card-customers')


st.sidebar.subheader('Controls')

show_raw_data = st.sidebar.checkbox('Show raw data')
show_target = st.sidebar.checkbox('Show Target')
show_variables = st.sidebar.checkbox('Choose variables')




if show_raw_data:
    st.subheader('Raw data')
    st.write(df)


if show_target:
    st.write(pie(static_df,"Attrition_Flag",["#4c758c","#e66c96"]))


if show_variables:

    st.sidebar.write("Choose two variables")
    multi_chosen = st.sidebar.multiselect('Select your vehicle:', df.columns)

    if len(multi_chosen) == 2:
        st.write(scatter(df,multi_chosen[0],multi_chosen[1]))