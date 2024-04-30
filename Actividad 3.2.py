import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Limpieza de datos nulos

df2020= pd.read_excel('gastos_costos_20_23.xlsx', sheet_name='2020')
df2021=pd.read_excel('gastos_costos_20_23.xlsx', sheet_name='2021')
df2022=pd.read_excel('gastos_costos_20_23.xlsx', sheet_name='2022')
df2023=pd.read_excel('gastos_costos_20_23.xlsx', sheet_name='2023')

df2020.info()

print(df2020.head())

def pocentajesNan(df):
    for i in df:
        if df[i].isnull().sum()/df.shape[0] > .5:
            df.drop(i, axis=1, inplace=True)
    return df 

nuevo2020=pocentajesNan(df2020)
nuevo2020.info()
print(nuevo2020.isnull().sum())

def rellenar_nulos(df):
    for i in df:
        if df[i].dtype=='int64' or df[i].dtype=='float64':
            df[i].fillna(df[i].mean(),inplace=True)

    df.dropna(inplace=True)
    return df

nuevo2020= rellenar_nulos(nuevo2020)
print(nuevo2020.isnull().sum())

def nulos(df):
    a=pocentajesNan(df)
    b=rellenar_nulos(a)
    return b

nuevo2020=nulos(df2020)
nuevo2021=nulos(df2021)
nuevo2022=nulos(df2022)
nuevo2023=nulos(df2023)

#Limpieza de datos atípicos 

fig, ax = plt.subplots()
ax.boxplot([nuevo2020['IVA'],nuevo2021['IVA'],nuevo2022['IVA'],nuevo2023['IVA']])
ax.set_xticklabels(['2020', '2021', '2022', '2023'])
ax.set_title('IVA')
ax.set_ylabel('Valor')
plt.show()

fig, ax = plt.subplots()
ax.boxplot([nuevo2020['TOTAL SAT'],nuevo2021['TOTAL SAT'],nuevo2022['TOTAL SAT'],nuevo2023['TOTAL SAT']])
ax.set_xticklabels(['2020', '2021', '2022', '2023'])
ax.set_title('TOTAL SAT')
ax.set_ylabel('Valor')
plt.show()

fig, ax = plt.subplots()
ax.boxplot([nuevo2020['TOTAL MX'],nuevo2021['TOTAL MX'],nuevo2022['TOTAL MX'],nuevo2023['TOTAL MX']])
ax.set_xticklabels(['2020', '2021', '2022', '2023'])
ax.set_title('TOTAL MX')
ax.set_ylabel('Valor')
plt.show()

fig, ax = plt.subplots()
ax.hist(nuevo2020['IVA'], bins=10, alpha=0.5, label='2020')
ax.hist(nuevo2021['IVA'], bins=10, alpha=0.5, label='2021')
ax.hist(nuevo2022['IVA'], bins=10, alpha=0.5, label='2022')
ax.hist(nuevo2023['IVA'], bins=10, alpha=0.5, label='2023')
ax.set_title('IVA')
ax.set_ylabel('Frecuencia')
ax.set_xlabel('Valor')
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.hist(nuevo2020['TOTAL SAT'], bins=10, alpha=0.5, label='2020')
ax.hist(nuevo2021['TOTAL SAT'], bins=10, alpha=0.5, label='2021')
ax.hist(nuevo2022['TOTAL SAT'], bins=10, alpha=0.5, label='2022')
ax.hist(nuevo2023['TOTAL SAT'], bins=10, alpha=0.5, label='2023')
ax.set_title('TOTAL SAT')
ax.set_ylabel('Frecuencia')
ax.set_xlabel('Valor')
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.hist(nuevo2020['TOTAL MX'], bins=10, alpha=0.5, label='2020')
ax.hist(nuevo2021['TOTAL MX'], bins=10, alpha=0.5, label='2021')
ax.hist(nuevo2022['TOTAL MX'], bins=10, alpha=0.5, label='2022')
ax.hist(nuevo2023['TOTAL MX'], bins=10, alpha=0.5, label='2023')
ax.set_title('TOTAL MX')
ax.set_ylabel('Frecuencia')
ax.set_xlabel('Valor')
ax.legend()
plt.show()


#Limpieza columna de IVA 
q1 = nuevo2020['IVA'].quantile(0.25)
q3 = nuevo2020['IVA'].quantile(0.75)
iqr = q3 - q1
df = nuevo2020[(nuevo2020['IVA'] >= q1 - 1.5*iqr) & (nuevo2020['IVA'] <= q3 + 1.5*iqr)]
print('DataFrame name: ', '2020', 'Columna: ', 'IVA')
print('limite superior: ',  q3 + 1.5*iqr)
print('limite inferior: ',  q1 - 1.5*iqr)

q1 = nuevo2021['IVA'].quantile(0.25)
q3 = nuevo2021['IVA'].quantile(0.75)
iqr = q3 - q1
df = nuevo2021[(nuevo2021['IVA'] >= q1 - 1.5*iqr) & (nuevo2021['IVA'] <= q3 + 1.5*iqr)]
print('DataFrame name: ', '2021', 'Columna: ', 'IVA')
print('limite superior: ',  q3 + 1.5*iqr)
print('limite inferior: ',  q1 - 1.5*iqr)

q1 = nuevo2022['IVA'].quantile(0.25)
q3 = nuevo2022['IVA'].quantile(0.75)
iqr = q3 - q1
df = nuevo2022[(nuevo2022['IVA'] >= q1 - 1.5*iqr) & (nuevo2022['IVA'] <= q3 + 1.5*iqr)]
print('DataFrame name: ', '2022', 'Columna: ', 'IVA')
print('limite superior: ',  q3 + 1.5*iqr)
print('limite inferior: ',  q1 - 1.5*iqr)

q1 = nuevo2023['IVA'].quantile(0.25)
q3 = nuevo2023['IVA'].quantile(0.75)
iqr = q3 - q1
df = nuevo2023[(nuevo2023['IVA'] >= q1 - 1.5*iqr) & (nuevo2023['IVA'] <= q3 + 1.5*iqr)]
print('DataFrame name: ', '2023', 'Columna: ', 'IVA')
print('limite superior: ',  q3 + 1.5*iqr)
print('limite inferior: ',  q1 - 1.5*iqr)

#Limpieza columna de TOTAL SAT
q1 = nuevo2020['TOTAL SAT'].quantile(0.25)
q3 = nuevo2020['TOTAL SAT'].quantile(0.75)
iqr = q3 - q1
df = nuevo2020[(nuevo2020['TOTAL SAT'] >= q1 - 1.5*iqr) & (nuevo2020['TOTAL SAT'] <= q3 + 1.5*iqr)]
print('DataFrame name: ', '2020', 'Columna: ', 'TOTAL SAT')
print('limite superior: ',  q3 + 1.5*iqr)
print('limite inferior: ',  q1 - 1.5*iqr)

q1 = nuevo2021['TOTAL SAT'].quantile(0.25)
q3 = nuevo2021['TOTAL SAT'].quantile(0.75)
iqr = q3 - q1
df = nuevo2021[(nuevo2021['TOTAL SAT'] >= q1 - 1.5*iqr) & (nuevo2021['TOTAL SAT'] <= q3 + 1.5*iqr)]
print('DataFrame name: ', '2021', 'Columna: ', 'TOTAL SAT')
print('limite superior: ',  q3 + 1.5*iqr)
print('limite inferior: ',  q1 - 1.5*iqr)

q1 = nuevo2022['TOTAL SAT'].quantile(0.25)
q3 = nuevo2022['TOTAL SAT'].quantile(0.75)
iqr = q3 - q1
df = nuevo2022[(nuevo2022['TOTAL SAT'] >= q1 - 1.5*iqr) & (nuevo2022['TOTAL SAT'] <= q3 + 1.5*iqr)]
print('DataFrame name: ', '2022', 'Columna: ', 'TOTAL SAT')
print('limite superior: ',  q3 + 1.5*iqr)
print('limite inferior: ',  q1 - 1.5*iqr)

q1 = nuevo2023['TOTAL SAT'].quantile(0.25)
q3 = nuevo2023['TOTAL SAT'].quantile(0.75)
iqr = q3 - q1
df = nuevo2023[(nuevo2023['TOTAL SAT'] >= q1 - 1.5*iqr) & (nuevo2023['TOTAL SAT'] <= q3 + 1.5*iqr)]
print('DataFrame name: ', '2023', 'Columna: ', 'TOTAL SAT')
print('limite superior: ',  q3 + 1.5*iqr)
print('limite inferior: ',  q1 - 1.5*iqr)

#Limpieza columna de TOTAL MX

q1 = nuevo2020['TOTAL MX'].quantile(0.25)
q3 = nuevo2020['TOTAL MX'].quantile(0.75)
iqr = q3 - q1
df = nuevo2020[(nuevo2020['TOTAL MX'] >= q1 - 1.5*iqr) & (nuevo2020['TOTAL MX'] <= q3 + 1.5*iqr)]
print('DataFrame name: ', '2020', 'Columna: ', 'TOTAL MX')
print('limite superior: ',  q3 + 1.5*iqr)
print('limite inferior: ',  q1 - 1.5*iqr)

q1 = nuevo2021['TOTAL MX'].quantile(0.25)
q3 = nuevo2021['TOTAL MX'].quantile(0.75)
iqr = q3 - q1
df = nuevo2021[(nuevo2021['TOTAL MX'] >= q1 - 1.5*iqr) & (nuevo2021['TOTAL MX'] <= q3 + 1.5*iqr)]
print('DataFrame name: ', '2021', 'Columna: ', 'TOTAL MX')
print('limite superior: ',  q3 + 1.5*iqr)
print('limite inferior: ',  q1 - 1.5*iqr)

q1 = nuevo2022['TOTAL MX'].quantile(0.25)
q3 = nuevo2022['TOTAL MX'].quantile(0.75)
iqr = q3 - q1
df = nuevo2022[(nuevo2022['TOTAL MX'] >= q1 - 1.5*iqr) & (nuevo2022['TOTAL MX'] <= q3 + 1.5*iqr)]
print('DataFrame name: ', '2022', 'Columna: ', 'TOTAL MX')
print('limite superior: ',  q3 + 1.5*iqr)
print('limite inferior: ',  q1 - 1.5*iqr)

q1 = nuevo2023['TOTAL MX'].quantile(0.25)
q3 = nuevo2023['TOTAL MX'].quantile(0.75)
iqr = q3 - q1
df = nuevo2023[(nuevo2023['TOTAL MX'] >= q1 - 1.5*iqr) & (nuevo2023['TOTAL MX'] <= q3 + 1.5*iqr)]
print('DataFrame name: ', '2023', 'Columna: ', 'TOTAL MX')
print('limite superior: ',  q3 + 1.5*iqr)
print('limite inferior: ',  q1 - 1.5*iqr)

#Revisión de rangos 


fig, ax = plt.subplots()
ax.boxplot([nuevo2020['IVA'],nuevo2021['IVA'],nuevo2022['IVA'],nuevo2023['IVA']])
ax.set_xticklabels(['2020', '2021', '2022', '2023'])
ax.set_title('IVA')
ax.set_ylabel('Valor')
plt.show()

fig, ax = plt.subplots()
ax.boxplot([nuevo2020['TOTAL SAT'],nuevo2021['TOTAL SAT'],nuevo2022['TOTAL SAT'],nuevo2023['TOTAL SAT']])
ax.set_xticklabels(['2020', '2021', '2022', '2023'])
ax.set_title('TOTAL SAT')
ax.set_ylabel('Valor')
plt.show()

fig, ax = plt.subplots()
ax.boxplot([nuevo2020['TOTAL MX'],nuevo2021['TOTAL MX'],nuevo2022['TOTAL MX'],nuevo2023['TOTAL MX']])
ax.set_xticklabels(['2020', '2021', '2022', '2023'])
ax.set_title('TOTAL MX')
ax.set_ylabel('Valor')
plt.show()

nuevo2020.to_csv('limpio_2020.csv')
nuevo2021.to_csv('limpio_2021.csv')
nuevo2022.to_csv('limpio_2022.csv')
nuevo2023.to_csv('limpio_2023.csv')