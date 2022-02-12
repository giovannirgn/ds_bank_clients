import matplotlib.pyplot as plt
import io
from collections import Counter,OrderedDict


def pie(df,col,clrs):

    dim = Counter(df[col])
    dim = OrderedDict(dim.most_common())

    fig = plt.figure(figsize = (10,6))
    plt.title(col,fontsize =20)

    plt.pie(dim.values(), labels=dim.keys(), autopct='%1.1f%%',pctdistance=0.55,shadow=False,
            startangle=90,colors = clrs,radius=1,wedgeprops=dict(width=0.3, edgecolor='w'))

    plt.axis('equal')

    return fig


def scatter(df,col_1,col_2):

    fig = plt.figure(figsize = (10,6))
    plt.title(col_1 + " VS " + col_2, fontsize=20)
    plt.scatter(df[col_1],df[col_2],c = df["Attrition_Flag"])
    plt.legend()
    plt.xlabel(col_1)
    plt.ylabel(col_2)

    return fig




def describe(df, target):

    stats = df.describe().T
    group_by_group = df.groupby(by=target)
    mean_by_groups = group_by_attrition.mean().T

    return stats, mean_by_groups



def info(df):

    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()

    return s

