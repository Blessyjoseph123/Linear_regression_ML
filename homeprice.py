import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df=pd.read_csv("homeprice.csv")
print(df)

plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='+')
plt.show()


reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)
result_pred=reg.predict([[3300]])
print(result_pred)
print(reg.coef_)
print(reg.intercept_)


