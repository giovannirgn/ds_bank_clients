import numpy as np
import pandas as pd
import streamlit as st
import inspect
from sklearn.ensemble import RandomForestClassifier
import pickle

from graph_functions import pie, scatter, info, describe, mean_graph, plot_density, corr_heatmap
from model_functions import random_sample,iter_model,select_random_index, predict
from some_functions import type_of_attribute, get_correlation
from categorical_fetures_dict import features_dic,text1, text2


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

st.sidebar.subheader('Controls')
explore_checkbok = st.sidebar.checkbox('Expolre dataset')





if explore_checkbok:

    show_raw_data = st.sidebar.checkbox('Show raw data')

    graph_checkbox  = st.sidebar.checkbox('Visualization')


    if show_raw_data:


        st.subheader('Raw data')
        st.write(pre_replacement)

        show_info = st.checkbox('Info')

        if show_info:

            st.text(info(pre_replacement))

        show_features = st.checkbox('Categorical attributes encoding')

        if show_features:

            st.json(features_dic)


        show_target = st.checkbox('Show Target')


        if show_target:
            st.write(pie(static_df,"Attrition_Flag",["#4c758c","#e66c96"]))


    if graph_checkbox:


        barchart_checkbox = st.sidebar.checkbox('Barchart')

        if barchart_checkbox:

            stats, mean_by_groups = describe(pre_replacement, "Attrition_Flag")

            st.write(mean_graph(stats, mean_by_groups, [0, 2, 9], [6, 7, 8], [1, 3, 4, 5]))

            st.write(text1)


        scatter_checkbox = st.sidebar.checkbox('Scatterplot')

        if scatter_checkbox:

            if graph_checkbox:

                st.write("Choose two variables")

                multi_chosen = st.multiselect('Variables:', df.columns)

                if len(multi_chosen) == 2:

                    st.write(scatter(df,multi_chosen[0],multi_chosen[1]))

                    st.write()


        density_checkbox = st.sidebar.checkbox('Density')

        if density_checkbox:

            st.sidebar.write("Choose the variable")

            numeric_features, label_features = type_of_attribute(pre_replacement)

            variable_density = st.selectbox("List of numeric variables",pre_replacement[numeric_features].columns.drop("Attrition_Flag"))

            st.write(plot_density(pre_replacement[numeric_features],variable_density,"Attrition_Flag"))



        correlation_checkbox = st.sidebar.checkbox('Correlation')

        if correlation_checkbox:

            st.write(corr_heatmap(df))

            st.write(text2)

            st.write(get_correlation(df,"Attrition_Flag"))




model_checkbox = st.sidebar.checkbox('Model')

if model_checkbox:

    random_checkbox = st.sidebar.checkbox('Random sample')

    if random_checkbox:



        show_code = st.checkbox('Code')

        if show_code:

            st.code(inspect.getsource(random_sample),language="python")


    test_checkbox = st.sidebar.checkbox('Test')


    if test_checkbox:


        st.write("Choose attributes and target")

        att_chosen = st.multiselect('Variables:', df.columns)

        m_df = df[att_chosen].copy()

        iteration_box = st.number_input('Number of training iteration')

        number_existing_customers_box = st.number_input('Number of existing customers')

        number_attirited_customer_box = st.number_input('Number of attrited customers')


        training_button = st.button("Train")

        if training_button:


            with st.spinner("Learning..."):

                random_forest = RandomForestClassifier(random_state=42)

                acc = []

                happy_predict_really_happy = []
                angry_predict_really_angry = []

                for _ in range(int(iteration_box)):




                    accuracy, confusion_mat = iter_model(m_df,


                                                         "Attrition_Flag",
                                                         int(number_existing_customers_box),
                                                         int(number_attirited_customer_box),
                                                         random_forest)

                    acc.append(accuracy)


                    true_negative = confusion_mat[0][0] / (confusion_mat[1][0] + confusion_mat[0][0])
                    true_positive = confusion_mat[1][1] / (confusion_mat[1][1] + confusion_mat[0][1])

                    happy_predict_really_happy.append(true_negative)
                    angry_predict_really_angry.append(true_positive)



                st.write(f"Mean of the accuracies: {np.array(acc).mean()}")


                st.write(f"Mean of the percentage  right 0 predictions:; {np.array(happy_predict_really_happy).mean()}")


                st.write(f"Mean of the percentage  right 1 predictions: {np.array(angry_predict_really_angry).mean()}")

                pickle.dump(random_forest, open("model", 'wb'))



    random_row_button = st.checkbox("Select random clients")

    model  = pickle.load(open("model", 'rb'))

    if random_row_button:

        sample_df, x, y = select_random_index(df,att_chosen,"Attrition_Flag")

        counter = 0
        for test_ in x:

            pred = predict(test_.reshape(1,-1),model)

            st.write(f"Client ID {sample_df.index[counter]}")
            st.write(f"Predict: {pred}")
            st.write(f"Actual: {y[counter]}")
            counter +=1


























