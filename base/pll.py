import numpy as np
import random as rdm
import pandas as pd
import matplotlib.pyplot as pl
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#test
#n = 10000
# Cargas los digitos de prueba
#x = [r for r in range(n)]
#y = [200,300,100,500,1021,1029,665]
#y = [int(rdm.random() * 500) + r * 6  for r in range(n)]

x = [1,2,3,4,5,6,7]
y = [200,300,100,500,1021,1029,665]

info = pd.DataFrame({'days': x , 'users': y})

days = info['days'].values.reshape(-1,1)
user = info['users'].values.reshape(-1,1)

days_train = info['days'].values.reshape(-1,1)#[:-20]
days_set = info['days'].values.reshape(-1,1)#[-20:]

user_train = info['users'].values.reshape(-1,1)#[:-20]
user_set = info['users'].values.reshape(-1,1)#[-20:]

#days_train = info['days'].values.reshape(-1,1)[:-5000]
#days_set = info['days'].values.reshape(-1,1)[-5000:]

#user_train = info['users'].values.reshape(-1,1)[:-5000]
#user_set = info['users'].values.reshape(-1,1)[-5000:]

#Creo la variable de la regresion lineal
regs = linear_model.LinearRegression()

regs.fit(days_train,user_train)

y_predict = regs.predict(days_set)

# The coefficients
print("Coefficients: \n", regs.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(user_set, y_predict))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(user_set, y_predict))

#pl.scatter(x_train,y_train,color='r')
pl.plot(days_set,user_set,'o',color= 'b', label='Usuarios/ dias')
pl.plot(days_set,y_predict,color= 'g',label='Regresion')
pl.xlabel('Dias de la semana')
pl.ylabel('Usuarios')
pl.title('Ajuste de datos')
pl.grid()
pl.legend(loc=4)
pl.show()







