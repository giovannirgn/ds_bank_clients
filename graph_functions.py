import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
    mean_by_groups = group_by_group.mean().T

    return stats, mean_by_groups



def info(df):

    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()

    return s

def groupby_target_graph(stats,mean_by_groups,index,i):

    mean_color = "#939597"
    att_color = "#f54242"
    ex_color = "#a1e4f7"

    labels = pd.Series(stats["mean"].index).iloc[index]

    mean_value = stats["mean"].iloc[index]

    att_value = mean_by_groups['Attrited Customer'].iloc[index]
    exi_value = mean_by_groups['Existing Customer'].iloc[index]

    x = np.arange(len(labels))

    width = 0.20

    sub = plt.subplot(1, 3, i)


    rects1 = plt.bar(x - width, att_value, width, label='Attrited Customer', color=att_color,edgecolor = "w", linewidth=4 )
    rects2 = plt.bar(x, mean_value, width, label='Sample Mean', color=mean_color,edgecolor = "w",linewidth=4 )
    rects3 = plt.bar(x + width, exi_value, width, label='Existing Customer', color=ex_color,edgecolor = "w",linewidth=4)



    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=90, fontsize=12)

    return sub ,




def mean_graph(stats, mean_by_groups,index_1,index_2,index_3):


    index = [index_1,index_2,index_3]

    fig = plt.figure(figsize=(13, 8))



    fig.suptitle("Differences between the means", fontsize=16)

    for i in range(3):

        groupby_target_graph(stats, mean_by_groups, index[i],i+1)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)


    return fig





