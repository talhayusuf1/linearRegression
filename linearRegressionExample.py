import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# bosluklşarı doldurmak için kullandık
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression  # Lineer regresion
veriler = pd.read_csv("satislar.csv")

aylar = veriler[["Aylar"]]
print(aylar)

satislar = veriler[["Satislar"]]
print(satislar)

satislar2 = veriler.iloc[:, :1].values
print(satislar2)

# aylar bagımsız satislar baglı degisken
x_train, x_test, y_train, y_test = train_test_split(
    aylar, satislar, test_size=0.33, random_state=0)
"""
sc = StandardScaler()

X_train = sc.fit_transform(x_train)

X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

"""
# Model İnsaasi(Linear Regresion)
lr = LinearRegression()
lr.fit(x_train, y_train)
# tahmin
tahmin = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()
print(x_train)
print(y_train)


plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))
plt.title("Aylara gore satis")
plt.xlabel("Aylar")
plt.ylabel("Satislar")
plt.show()
