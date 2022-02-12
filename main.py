import numpy as np
import pandas as pd
import streamlit as st

from categorical_fetures_dict import features_dic ,text1
from graph_functions import pie, scatter, info, describe, mean_graph



static_df = pd.read_csv("dataset.csv",sep=",")

df = static_df.copy()
df.set_index("CLIENTNUM",inplace=True,drop=True)
df = df.iloc[:,:-4].drop(["Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1"],axis=1)
pre_replacement = df.copy()
df.replace(features_dic,inplace=True)
df["Income_Category"].replace(0,df["Income_Category"].mean(),inplace=True)






st.header('Clients Classification')
st.write("Who will close the account?")
st.write('Dataset source: www.kaggle.com/sakshigoyal7/credit-card-customers')

stats, mean_by_groups = describe(pre_replacement,"Attrition_Flag")








st.sidebar.subheader('Controls')
explore_checkbok = st.sidebar.checkbox('Expolre dataset')





if explore_checkbok:

    show_raw_data = st.sidebar.checkbox('Show raw data')

    graph_checkbox  = st.sidebar.checkbox('Visualization')


    if show_raw_data:


        st.subheader('Raw data')
        st.write(df)

        show_info = st.checkbox('Info')

        if show_info:
            st.text(info(df))

        show_features = st.checkbox('Categorical attributes encoding')

        if show_features:
            st.write(features_dic)


        show_target = st.checkbox('Show Target')


        if show_target:
            st.write(pie(static_df,"Attrition_Flag",["#4c758c","#e66c96"]))


    if graph_checkbox:

        barchart_checkbox = st.sidebar.checkbox('Barchart')

        if barchart_checkbox:

            st.write(mean_graph(stats, mean_by_groups, [0, 2, 9], [6, 7, 8], [1, 3, 4, 5]))

            st.write(text1)

        scatter_checkbox = st.sidebar.checkbox('Scatterplot')

        if scatter_checkbox:

            if graph_checkbox:

                st.sidebar.write("Choose two variables")
                multi_chosen = st.sidebar.multiselect('Variables:', df.columns)

                if len(multi_chosen) == 2:

                    st.write(scatter(df,multi_chosen[0],multi_chosen[1]))

                    st.write()





model_checkbox = st.sidebar.checkbox('Model')