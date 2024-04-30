import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#print('hello outliers')
df= pd.read_csv('ventas_totales_sinnulos.csv', index_col=0)
#print(df.head())

valores_nulos=df.isnull().sum()

print(valores_nulos)

#### Limpieza de columna ventas_precios_constantes

fig = plt.figure(figsize =(7, 3))
plt.hist(x=df["ventas_precios_constantes"], color='red', rwidth=0.50)
plt.title('Histograma de ventas_precios_constantes con outliers')
plt.xlabel('ventas_precios_constantes')
plt.ylabel('Frecuencia')
#plt.show()

fig = plt.figure(figsize =(5, 3))
plt.boxplot(df["ventas_precios_constantes"]) 
plt.title("Outliers de ventas_precios_constantes")
#plt.show()

#Método aplicando Cuartiles. Encuentro cuartiles 0.25 y 0.75
y=df["ventas_precios_constantes"]
print(y)

percentile25=y.quantile(0.25) #Q1
percentile75=y.quantile(0.75) #Q3
print(percentile25)
print(percentile75)
iqr= percentile75 - percentile25
print(iqr)

Limite_Superior_iqr= percentile75 + 1.5*iqr
Limite_Inferior_iqr= percentile25 - 1.5*iqr
print("Limite superior permitido", Limite_Superior_iqr)
print("Limite inferior permitido", Limite_Inferior_iqr)

#Obtenemos datos limpios
data_clean_iqr= df[ (y< Limite_Superior_iqr) & (y > Limite_Inferior_iqr) ]
print(data_clean_iqr)

fig = plt.figure(figsize =(5, 3))
plt.boxplot(data_clean_iqr["ventas_precios_constantes"]) 
plt.title("Outliers de ventas_precios_constantes")
plt.show()

fig = plt.figure(figsize =(7, 3))
plt.hist(x=data_clean_iqr["ventas_precios_constantes"], color='blue', rwidth=0.50)
plt.title('Histograma de ventas_precios_constantes sin outliers')
plt.xlabel('ventas_precios_constantes')
plt.ylabel('Frecuencia')
plt.show()

#data_clean_iqr['ventas_precios_constantes'].to_csv('ventas_precios_constantes.csv')


#### Limpieza de columna salon_ventas

fig = plt.figure(figsize =(7, 3))
plt.hist(x=df["salon_ventas"], color='red', rwidth=0.50)
plt.title('Histograma de salon_ventas con outliers')
plt.xlabel('salon_ventas')
plt.ylabel('Frecuencia')
#plt.show()

fig = plt.figure(figsize =(5, 3))
plt.boxplot(df["salon_ventas"]) 
plt.title("Outliers de salon_ventas")
#plt.show()

#Método aplicando Cuartiles. Encuentro cuartiles 0.25 y 0.75
y=df["salon_ventas"]
print(y)

percentile25=y.quantile(0.25) #Q1
percentile75=y.quantile(0.75) #Q3
print(percentile25)
print(percentile75)
iqr= percentile75 - percentile25
print(iqr)

Limite_Superior_iqr= percentile75 + 1.5*iqr
Limite_Inferior_iqr= percentile25 - 1.5*iqr
print("Limite superior permitido", Limite_Superior_iqr)
print("Limite inferior permitido", Limite_Inferior_iqr)

#Obtenemos datos limpios
data_clean_iqr= df[ (y< Limite_Superior_iqr) & (y > Limite_Inferior_iqr) ]
print(data_clean_iqr)

fig = plt.figure(figsize =(5, 3))
plt.boxplot(data_clean_iqr["salon_ventas"]) 
plt.title("Outliers de salon_ventas")
plt.show()

fig = plt.figure(figsize =(7, 3))
plt.hist(x=data_clean_iqr["salon_ventas"], color='blue', rwidth=0.50)
plt.title('Histograma de salon_ventas sin outliers')
plt.xlabel('salon_ventas')
plt.ylabel('Frecuencia')
plt.show()

#data_clean_iqr['salon_ventas'].to_csv('salon_ventas_limpios.csv')

#Limpieza de columna efectivo

fig = plt.figure(figsize =(7, 3))
plt.hist(x=df["efectivo"], color='red', rwidth=0.50)
plt.title('Histograma de efectivo con outliers')
plt.xlabel('efectivo')
plt.ylabel('Frecuencia')
#plt.show()

fig = plt.figure(figsize =(5, 3))
plt.boxplot(df["efectivo"]) 
plt.title("Outliers de efectivo")
#plt.show()

#Método aplicando Cuartiles. Encuentro cuartiles 0.25 y 0.75
y=df["efectivo"]
print(y)

percentile25=y.quantile(0.25) #Q1
percentile75=y.quantile(0.75) #Q3
print(percentile25)
print(percentile75)
iqr= percentile75 - percentile25
print(iqr)

Limite_Superior_iqr= percentile75 + 1.5*iqr
Limite_Inferior_iqr= percentile25 - 1.5*iqr
print("Limite superior permitido", Limite_Superior_iqr)
print("Limite inferior permitido", Limite_Inferior_iqr)

#Obtenemos datos limpios
data_clean_iqr= df[ (y< Limite_Superior_iqr) & (y > Limite_Inferior_iqr) ]
print(data_clean_iqr)

fig = plt.figure(figsize =(5, 3))
plt.boxplot(data_clean_iqr["efectivo"]) 
plt.title("Outliers de efectivo")
plt.show()

fig = plt.figure(figsize =(7, 3))
plt.hist(x=data_clean_iqr["efectivo"], color='blue', rwidth=0.50)
plt.title('Histograma de efectivo sin outliers')
plt.xlabel('efectivo')
plt.ylabel('Frecuencia')
plt.show()

data_clean_iqr['efectivo'].to_csv('efectivo_limpios.csv')

