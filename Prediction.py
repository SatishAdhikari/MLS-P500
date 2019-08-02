#All Imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import datetime
import pandas_datareader.data as web
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

pd.options.display.max_columns = 500
pd.options.display.max_rows = 500

df1 = pd.read_csv('GSPC.csv', parse_dates = ['Date']) #loading CSV
df1 = df1.drop(columns = ['Open','High', 'Low', 'Close', 'Volume']) #Removing uneeded info
df1 = df1.dropna()

wdw= 30
df1['Moving Average'] = df1['Adj Close'].rolling(window=wdw).mean() # Adding moving average
df1 = df1.dropna().reset_index().drop(columns = 'index')
# Calculating slope
FD = []
FD1 = []
n = 1#Days
for i in range(1,len(df1)):
    first_D = ((df1.loc[i,'Moving Average'] - df1.loc[i-n,'Moving Average'])/ n)#Rise over run
    FD.append(first_D)

for i in range(1,len(df1)):
    first_D1 = ((df1.loc[i,'Adj Close'] - df1.loc[i-n,'Adj Close'])/ n) #Rise over run
    FD1.append(first_D1)

df1 = df1.drop(index = list(range(n))).reset_index().drop(columns = 'index') # Removing the last n rows because i+n row will not exist

df1['FD Adj'] = np.transpose(FD1) #slope of original
df1['FD MA'] = np.transpose(FD) #Slope of moving average

SD = []
SD1 = []
for i in range(1,len(df1)):
    Second_D = ((df1.loc[i,'FD MA'] - df1.loc[i-n,'FD MA'])/ n)#Rise over run
    SD.append(Second_D)

for i in range(1,len(df1)):
    Second_D1 = ((df1.loc[i,'FD Adj'] - df1.loc[i-n,'FD Adj'])/ n)#Rise over run
    SD1.append(Second_D1)

df1 = df1.drop(index = list(range(n))).reset_index().drop(columns = 'index') # Removing the last n rows because i+n row will not exist

df1['SD Adj'] = np.transpose(SD1)
df1['SD MA'] = np.transpose(SD)

#df1 = df1.loc[(df1['Date'] > '2015-01-01') & (df1['Date'] < '2016-01-01')] # Creating a date range
df1 = df1.reset_index()
Day_1 = df1.drop(columns = ['Date','Adj Close','Moving Average','index'])
Day_2 = Day_1.drop(index = [0]).reset_index()
Day_3 = Day_1.drop(index = [0,1]).reset_index()
Day_4 = Day_1.drop(index = [0,1,2]).reset_index()
Day_5 = Day_1.drop(index = [0,1,2,3]).reset_index()
Day_2 = Day_2.drop(columns = ['index'])
Day_3 = Day_3.drop(columns = ['index'])
Day_4 = Day_4.drop(columns = ['index'])
Day_5 = Day_5.drop(columns = ['FD MA','SD MA','SD Adj','index']) # The amount of money gained from the previous day
Day_5_bi = Day_5['FD Adj'].apply(lambda x: 1 if x > 0 else 0)


previous_days = pd.concat([Day_1, Day_2, Day_3, Day_4], axis=1, sort=False)
previous_days = previous_days.drop(index = [0,1,2,3,4,5]).reset_index() # Five days of information on FD and SD
training_points = previous_days.drop(columns = ['index'])
training_labels = Day_5_bi

training_points = training_points.dropna()
training_labels = training_labels.dropna()
training_labels = training_labels.drop(index = list(range(len(training_labels)-5,len(training_labels))))
print(len(training_points), len(training_labels), training_points, training_labels)
print(df1.head(15))
print(len(df1))
