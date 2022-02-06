import pandas as pd

df = pd.read_csv("dataset.csv",sep=",")
df.set_index("CLIENTNUM",inplace=True,drop=True)
df = df.iloc[:,:-4].drop(["Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1"],axis=1)


print(df.columns)