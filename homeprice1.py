import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df=pd.read_csv("homeprice.csv")
print(df)




reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)
result_pred=reg.predict(df[['area']])
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='+')
plt.plot(df.area,result_pred,color='blue',marker='o')
plt.show()

