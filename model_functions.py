import pandas as pd
import numpy as np

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def random_set(df,target,size_x,size_y):

  ones_df = df[df[target]==1]
  zeros_df = df[df[target]==0]

  random_ones_index = np.random.choice(ones_df.index,size=size_x,replace=False)
  random_zeros_index = np.random.choice(zeros_df.index,size=size_y,replace=False)

  random_ones = ones_df.loc[random_ones_index]
  random_zeros = zeros_df.loc[random_zeros_index]

  concat_df = pd.concat([random_ones,random_zeros])

  i = 6

  random_df = shuffle(concat_df)

  while i >= 0:

    random_df = shuffle(random_df)

    i -=1


  return  random_df,


def features_and_target_arrays(df,target):

  features = df.drop(target,axis=1).values
  target = df[target].values

  return features,target



def train_test(x,y,test_size=0.2,random_state=1):

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state=random_state)

  return x_train, x_test, y_train, y_test


def train_model(x_train, y_train, x_test, y_test, model):

  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)

  return accuracy_score(y_pred, y_test), confusion_matrix(y_test, y_pred, labels=[0, 1])


def iter_model(df, target, zeros_size, ones_size, model):

  random = random_set(df, target, ones_size, zeros_size)

  X, y = features_and_target_arrays(random, target)

  x_train, x_test, y_train, y_test = train_test_split(X, y)

  accuracy, confusion_mat = train_model(x_train, y_train, x_test, y_test, model)

  return accuracy, confusion_mat
