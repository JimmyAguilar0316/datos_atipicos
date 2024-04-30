import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df2020= pd.read_excel('gastos_costos_20_23.xlsx', index_col=0)

print(df2020.head())

#valores_nulos=df2020.isnull().sum()

#print(valores_nulos)

#print(df2020.describe())

#print(df2020.info())

#print(df2020.head())

#df2020=df2020.dropna(subset=['FOLIO','GASTO','POLIZA'])

#valores_nulos=df2020.isnull().sum()

#print(valores_nulos)

df2020[['TC','IMPORTE','IVA']] =df2020[['TC','IMPORTE','IVA']].fillna(round(df2020[['TC','IMPORTE','IVA']].mean(),1))

valores_nulos=df2020.isnull().sum()
print(valores_nulos)

print(df2020)









