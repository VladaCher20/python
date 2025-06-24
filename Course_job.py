import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pylab 
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
import numpy as np
import datetime as dt

def ShowYPredicted(Y1,Y2,Title):
    Indexes = np.linspace(0, len(Y1)-1, len(Y1))
    plt.plot(Indexes, Y1,c = 'b')
    plt.plot(Indexes, Y2,c = 'r')
    plt.xlabel(' ') 
    plt.ylabel('Цены')
    plt.title(Title)
    plt.show()


def Mat_Exp(df,column_name):
    Summ = df[column_name].sum()
    num_rows = df[column_name].count()
    return (Summ/num_rows)

def BoxPlots(df):
    sns.boxplot(data=df, x="price", y="state")
    #sns.boxplot(x=df2["bed"])
    plt.title("Boxplot-график цен для каждого штата")
    plt.show()


def QQPlots(df):
    fig = sm.qqplot(df["bed"], line = "q")
    plt.title("qq_plot: bed")
    plt.show()
    fig = sm.qqplot(df["bath"], line = "q")
    plt.title("qq_plot: bath")
    plt.show()
    fig = sm.qqplot(df["acre_lot"], line = "q")
    plt.title("qq_plot: acre_lot")
    plt.show()
    fig = sm.qqplot(df["house_size"], line = "q")
    plt.title("qq_plot: house_size")
    plt.show()
    fig = sm.qqplot(df["price"], line = "q")
    plt.title("qq_plot: price")
    plt.show()


def PairPlots(df):
    plt.figure(figsize=(12,7))
    sns.scatterplot(data=df,x="state",y="price")
    plt.title("Зависимость цен от штатов")
    plt.show()
    plt.figure(figsize=(12,7))
    sns.scatterplot(data=df,x="bed",y="price")
    plt.title("Зависимость цен от количества кроватей")
    plt.show()
    plt.figure(figsize=(12,7))
    sns.scatterplot(data=df,x="bath",y="price")
    plt.title("Зависимость цен от количества уборных")
    plt.show()
    plt.figure(figsize=(12,7))
    sns.scatterplot(data=df,x="house_size",y="price")
    plt.title("Зависимость цен от размера дома")
    plt.show()
    plt.figure(figsize=(12,7))
    sns.scatterplot(data=df,x="acre_lot",y="price")
    plt.title("Зависимость цен от стоимости акра")
    plt.show()

  
df = pd.read_csv("Data.csv")





#Пункт 1
#1. Осмотр данных
print("\nВерхние строки таблицы:")
print(df.head())
print("\nРазмер таблицы:")
print(df.shape)
print("\nИнформация об столбцах таблицы:")
print(df.info())
print("\nИнформация об пустых значениях по столбцам:")
print(df.isnull().sum())
print("\nИнформация о дубликатах:")
print(df.duplicated().sum())
print("\nИнформация о данных таблицы ДО УДАЛЕНИЯ ДУБЛИКАТОВ:")
print(df.describe().round(2))
print("\nМат.ожидание столбца bed = ", Mat_Exp(df,"bed"))
print("\nМат.ожидание столбца bath = ", Mat_Exp(df,"bath"))
print("\nМат.ожидание столбца acre_lot = ", Mat_Exp(df,"acre_lot"))
print("\nМат.ожидание столбца zip_code = ", Mat_Exp(df,"zip_code"))
print("\nМат.ожидание столбца house_size = ", Mat_Exp(df,"house_size"))
print("\nМат.ожидание столбца price = ", Mat_Exp(df,"price"))
print("\nДисперсия\n\n",df.var(numeric_only=True))

#Матрица корреляции
print("\nМатрица корреляции\n", df.corr(numeric_only=True))

#Визуализация
#BoxPlots(df)
#QQPlots(df)
#PairPlots(df)

#Пункт 2
#2. Удаляем дубликаты
print("\nИнформация об пустых значениях по столбцам после удаления дубликатов:")
df.drop_duplicates(inplace=True)
print(df.isnull().sum())

print("\nИнформация о данных таблицы ПОСЛЕ УДАЛЕНИЯ ДУБЛИКАТОВ:")
print(df.describe().round(2))
print("\nМат.ожидание столбца bed = ", Mat_Exp(df,"bed"))
print("\nМат.ожидание столбца bath = ", Mat_Exp(df,"bath"))
print("\nМат.ожидание столбца acre_lot = ", Mat_Exp(df,"acre_lot"))
print("\nМат.ожидание столбца zip_code = ", Mat_Exp(df,"zip_code"))
print("\nМат.ожидание столбца house_size = ", Mat_Exp(df,"house_size"))
print("\nМат.ожидание столбца price = ", Mat_Exp(df,"price"))
print("\nДисперсия\n\n",df.var(numeric_only=True))
print("\nМатрица корреляции\n", df.corr(numeric_only=True))

#3.Очистка
#3.a Удаляем критерии 'status','city','zip_code','prev_sold_date' за ненадобностью
print("\nВерхние строки таблицы после удаления критериев:")
df2 = df.drop(['status','city','zip_code','prev_sold_date'],axis=1)
print(df2.head())
print(df2.shape)

print("\nИнформация об пустых значениях по столбцам после удаления критериев:")
print(df2.isnull().sum())

#3.b Заменяем пустые значения у критериев 'bed' и 'bath' на 3 и 2.
print("\nЗаменяем пустые значения у критериев bed и bath На 3 и 2.")
df2["bed"]=df2["bed"].fillna(3)
df2["bath"]=df2["bath"].fillna(2)

#3.c Удаляем строки с пустыми значениями по критериям 'price', 'house_size', 'acre_lot' - нельзя использовать такие для прогнозирования и нечем заменить.
print("\nУдаляем строки, в которых есть пустые значения по критериям price, house_size или acre_lot.")
df2.dropna(subset=["price","house_size","acre_lot"],inplace=True)

print("\nИнформация о данных таблицы ПОСЛЕ ОЧИСТКИ:")
print(df2.describe().round(2))
print("\nМат.ожидание столбца bed = ", Mat_Exp(df2,"bed"))
print("\nМат.ожидание столбца bath = ", Mat_Exp(df2,"bath"))
print("\nМат.ожидание столбца acre_lot = ", Mat_Exp(df2,"acre_lot"))
print("\nМат.ожидание столбца house_size = ", Mat_Exp(df2,"house_size"))
print("\nМат.ожидание столбца price = ", Mat_Exp(df2,"price"))
print("\nДисперсия\n\n",df2.var(numeric_only=True))
print("\nМатрица корреляции\n", df2.corr(numeric_only=True))


df2.drop([121247,1143422,80518,108951,1376991,10328,475143,1161246,734849],inplace=True)


print("\nОтсортированные данные по price:")
print(df2.sort_values(by="price",ascending=False).head())
print("\nОтсортированные данные по bed:")
print(df2.sort_values(by="bed",ascending=False).head())
print("\nОтсортированные данные по bath:")
print(df2.sort_values(by="bath",ascending=False).head())
print("\nОтсортированные данные по house_size:")
print(df2.sort_values(by="house_size",ascending=False).head())
print("\nОтсортированные данные по acre_lot:")
print(df2.sort_values(by="house_size",ascending=False).head())

#PairPlots(df2)

print("\nИнформация о данных таблицы после удаления выбросов:")
print(df2.describe().round(2))
print("\nМат.ожидание столбца bed = ", Mat_Exp(df2,"bed"))
print("\nМат.ожидание столбца bath = ", Mat_Exp(df2,"bath"))
print("\nМат.ожидание столбца acre_lot = ", Mat_Exp(df2,"acre_lot"))
print("\nМат.ожидание столбца house_size = ", Mat_Exp(df2,"house_size"))
print("\nМат.ожидание столбца price = ", Mat_Exp(df2,"price"))
print("\nДисперсия\n\n",df2.var(numeric_only=True))
print("\nМатрица корреляции\n", df2.corr(numeric_only=True))

print("\nИнформация о текущих данных:")
print(df2.isnull().sum())
print(df2.shape)
print(df2.head())
print(df2.info())


#Пункт 3
#4. Исследуем критерий 'state' - единственный оставшийся качественный критерий.
#4.a Сколько их всего
print("\nИнформация о всех различных значениях state:")
print(df2["state"].unique())

#4.b Средняя стоимость по каждому штату. Отсортировано.
print("\nСредняя стоимость по каждому штату:")
df3 = df2.groupby(['state']).mean()
#df3 = df3.drop(['bed','bath','acre_lot','house_size'],axis=1)
print(df3.sort_values(by="price",ascending=False))

for i, col in enumerate(['state']):
    sns.catplot(x=col,y='price',data=df2,kind='point',aspect=2,)
    plt.show()

#4.c Меняем признак state.


df2['state'].replace('Virgin Islands', '2081351', inplace=True)
df2['state'].replace('Massachusetts', '1109773', inplace=True)
df2['state'].replace('New York', '871346', inplace=True)
df2['state'].replace('Connecticut', '783611', inplace=True)
df2['state'].replace('Puerto Rico', '652912', inplace=True)
df2['state'].replace('New Hampshire', '640345', inplace=True)
df2['state'].replace('Rhode Island', '616646', inplace=True)
df2['state'].replace('New Jersey', '597498', inplace=True)
df2['state'].replace('Vermont', '554226', inplace=True)
df2['state'].replace('Wyoming', '535000', inplace=True)
df2['state'].replace('Maine', '511439', inplace=True)
df2['state'].replace('Pennsylvania', '427915', inplace=True)
df2['state'].replace('Delaware', '366114', inplace=True)
df2['state'].replace('West Virginia', '62500', inplace=True)
df2["state"]=pd.to_numeric(df2["state"])

print("\nИнформация о данных таблицы после изменения категориального признака:")
print(df2.describe().round(2))
print("\nМат.ожидание столбца bed = ", Mat_Exp(df2,"bed"))
print("\nМат.ожидание столбца bath = ", Mat_Exp(df2,"bath"))
print("\nМат.ожидание столбца acre_lot = ", Mat_Exp(df2,"acre_lot"))
print("\nМат.ожидание столбца house_size = ", Mat_Exp(df2,"house_size"))
print("\nМат.ожидание столбца price = ", Mat_Exp(df2,"price"))
print("\nДисперсия\n\n",df2.var(numeric_only=True))

print("\nИнформация о текущих данных:")
#print(df2.isnull().sum())
#print(df2.shape)
print(df2.head())
print(df2.info())
print("\nМатрица корреляции\n", df2.corr(numeric_only=True))
print('')



from sklearn.decomposition import PCA

prep1 = df2.drop(['state'],axis=1)
prep2 = df2.drop(['bed','bath','acre_lot','house_size','price'],axis=1)

X = prep1.values
Y = prep2.values

# Применение PCA для сокращения размерности
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Визуализация данных
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=Y, cmap='viridis')
plt.title("Визуализация данных с PCA")
plt.xlabel("Главная компонента 1")
plt.ylabel("Главная компонента 2")
plt.show()


#Пункт 4

#Пункт 4.1
#Случайный отбор
#Генерация случайной выборки

N = len(df2)
n = int(N * 0.3)
df = df2.sample(n)

print("Общее число генеральной совокупности N =",N)
print("")
print("n - 30% генеральной выборки =", n)

print("")
print(df)

#writer = pd.ExcelWriter('random.xlsx', engine='xlsxwriter')    
#df.to_excel(writer, 'Sheet1') 
#writer.save()





#Пункт 4.2
#Стратифицированный отбор
print('Стратифицированный отбор')
df = df2.groupby('state', group_keys=False).apply(lambda x: x.sample(frac=0.3))


#df = pd.read_excel("random.xlsx") #случайная выборка
#df.drop('id',axis=1, inplace=True)

#Повтор Пункт 1
#1. Осмотр данных
print("\nВерхние строки таблицы:")
print(df.head())
print("\nРазмер таблицы:")
print(df.shape)
print("\nИнформация об столбцах таблицы:")
print(df.info())
print("\nИнформация о данных таблицы:")
print(df.describe().round(2))
print("\nМат.ожидание столбца bed = ", Mat_Exp(df,"bed"))
print("\nМат.ожидание столбца bath = ", Mat_Exp(df,"bath"))
print("\nМат.ожидание столбца acre_lot = ", Mat_Exp(df,"acre_lot"))
print("\nМат.ожидание столбца house_size = ", Mat_Exp(df,"house_size"))
print("\nМат.ожидание столбца state = ", Mat_Exp(df,"state"))
print("\nМат.ожидание столбца price = ", Mat_Exp(df,"price"))
print("\nДисперсия\n\n",df.var(numeric_only=True))

#Матрица корреляции
print("\nМатрица корреляции\n", df.corr(numeric_only=True))



#Часть курсовой

#Визуальное представление матрицы корреляции
plt.figure(figsize=(12, 6))
sns.heatmap(df2.corr(),
            cmap = 'BrBG',
            fmt = '.2f',
            linewidths = 2,
            annot = True)
plt.show()



#Разделение набора данных на обучение и тестирование
X = df2.drop(['price'], axis=1) #все остальные стольцы, кроме price
Y = df2['price'] #столбец price


 
# Разделить обучающий набор на 
# обучающий и проверочный набор
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=0)




#SVM – машина опорных векторов
model_SVR = svm.SVR()
start = dt.datetime.now() 

model_SVR.fit(X_train,Y_train)
Y_pred = model_SVR.predict(X_test)
time = dt.datetime.now() - start

print('SVM:')
print('time = ',time)
print('MAE = ',mean_absolute_percentage_error(Y_test, Y_pred))
print('RMSE =', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
print('R2 = ', np.round(metrics.r2_score(Y_test, Y_pred), 2))
print('')


ShowYPredicted(Y_test,Y_pred,"SVM")



#Регрессия случайного леса
model_RFR = RandomForestRegressor(n_estimators=10)
start = dt.datetime.now() 
model_RFR.fit(X_train, Y_train)
Y_pred = model_RFR.predict(X_test)
time = dt.datetime.now() - start
print('RFR:')
print('time = ',time)
print('MAE = ',mean_absolute_percentage_error(Y_test, Y_pred)) 
print('RMSE =', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
print('R2 = ', np.round(metrics.r2_score(Y_test, Y_pred), 2))
print('')

ShowYPredicted(Y_test,Y_pred,"RandomForest")




#Линейная регрессия
# создадим объект этого класса и запишем в переменную model
model_LR = LinearRegression()
start = dt.datetime.now() 
model_LR.fit(X_train, Y_train) # обучим нашу модель
time = dt.datetime.now() - start
Y_pred = model_LR.predict(X_test) # на основе нескольких независимых переменных (Х) предскажем цену на жилье (y)
print('Линейная регрессия:') # MAE - Потеря регрессии с абсолютной ошибкой среднего значения.
print('time = ',time)
print('MAE = ',mean_absolute_percentage_error(Y_test, Y_pred))
print('RMSE =', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
print('R2 = ', np.round(metrics.r2_score(Y_test, Y_pred), 2))
print('')

ShowYPredicted(Y_test,Y_pred,"LinearRegression")



#CatBoostRegressor для регрессии - библиотека градиентного бустинга
cb_model = CatBoostRegressor()
start = dt.datetime.now() 
cb_model.fit(X_train, Y_train)
time = dt.datetime.now() - start
Y_pred = cb_model.predict(X_test) 
 
cb_r2_score=r2_score(Y_test, Y_pred)
print('CatBoost:')
print('time = ',time)
print('MAE = ',cb_r2_score)
print('RMSE =', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
print('R2 = ', np.round(metrics.r2_score(Y_test, Y_pred), 2))

ShowYPredicted(Y_test,Y_pred,"CatBoost")



