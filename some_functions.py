import pandas as pd


def type_of_attribute(df):

    numeric_features = ['Attrition_Flag']
    label_features = []

    for col in df.columns:

        if df[col].dtypes == "object":

            label_features.append(col)

        else:

            numeric_features.append(col)

    return numeric_features, label_features


def get_correlation(df, a):

    corr = df.corr()[a].sort_values(ascending=False)

    return corr