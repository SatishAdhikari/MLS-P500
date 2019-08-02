#All Imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import datetime
import pandas_datareader.data as web
import csv
from pandas.plotting import register_matplotlib_converters

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

register_matplotlib_converters()

pd.options.display.max_columns = 500
pd.options.display.max_rows = 500

df1 = pd.read_csv('DJI.csv', parse_dates = ['Date']) #loading CSV
df1 = df1.drop(columns = ['Open','High', 'Low', 'Close']) #Removing uneeded info
df1 = df1.dropna()

wdw = 19

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

#df1 = df1.loc[(df1['Date'] > '2010-01-01') & (df1['Date'] < '2018-02-01')] # Creating a date range

Day_1 = df1.reset_index().drop(columns = ['Date','Moving Average','index'])
Day_2 = Day_1.drop(index = [0]).reset_index().drop(columns = ['index'])
Day_3 = Day_1.drop(index = [0,1]).reset_index().drop(columns = ['index'])
Day_4 = Day_1.drop(index = [0,1,2]).reset_index().drop(columns = ['index'])
Day_5 = Day_1.drop(index = [0,1,2,3]).reset_index().drop(columns = ['index'])
Day_6 = Day_1.drop(index = [0,1,2,3,4]).reset_index().drop(columns = 'index')
Day_6 = Day_6.drop(columns = ['FD MA','SD MA','SD Adj','Volume']) # The amount of money gained from the previous day
Day_6['percent'] = (Day_6['FD Adj']/Day_6['Adj Close'])*100
Day_6 = Day_6.drop(columns = ['FD Adj', 'Adj Close']) # The amount of money gained from the previous day

p = 0

Day_6_bi = Day_6['percent'].apply(lambda x: 1 if x > p else 0)

percentage = Day_6_bi.isin([1]).sum(axis=0)/len(Day_6_bi)
print('Percent of days the S&P increased in value by more than {x} % is '.format(x = p), round(percentage * 100, 2), '%')

previous_days = pd.concat([Day_1, Day_2, Day_3, Day_4, Day_5], axis=1, sort=False)
training_points = previous_days.drop(index = list(range(len(previous_days)-5,len(previous_days)))).reset_index().drop(columns = ['index']) # Five days of information on FD and SD
training_labels = Day_6_bi

training_points = training_points.dropna()

training_labels = training_labels.dropna()
binary_comp = pd.concat([Day_6, Day_6_bi], axis = 1, sort = False)

training_points_2 = training_points.to_string()
training_labels_2 = training_labels.to_string()

#training_points.to_csv('training_points.csv', header = False)
#training_labels.to_csv('training_labels.csv', header = False)
with open('training_points.csv','a') as fd:
    fd.write(training_points_2)

with open('training_labels.csv','a') as fd:
    fd.write(training_labels_2)
'''
    scaler = MinMaxScaler()
    training_points_scaled = scaler.fit_transform(training_points)
    #print(training_points_scaled)

    #from Prediction import prediction
    X_train, X_test, y_train, y_test = train_test_split(training_points_scaled, training_labels, test_size=0.20, random_state=15) #42

    classifier = KNeighborsClassifier(n_neighbors= 97)
    classifier.fit(X_train, y_train)
    scores = classifier.score(X_test, y_test)
    cv_scores.append(round(scores*100,2))

plt.style.use('seaborn-poster')
plt.title('Window size versus accuracy')
plt.xlabel('Days in Moving average')
plt.ylabel('Percentage Accuracy')
plt.plot(window_range, cv_scores, color = 'r')
plt.show()
'''
#print(binary_comp.head(400))
#print(training_points.head(20))
print(len(training_points), len(training_labels))
#print(df1.head(15))
#print(len(df1))
#print(len(training_points))
#print(training_labels[])
'''
plt.style.use('seaborn-poster')

plt.title('Stock Price')
plt.ylabel('Price')
plt.xlabel('Dates')
plt.plot(df1.Date, df1['Adj Close'], color = 'r', label = 'Adj Close')
plt.plot(df1.Date, df1['Moving Average'], color = 'b', label = '{X} day Moving average'.format(X = wdw))
plt.legend()
plt.show()

plt.title('{D} day Moving slope'.format(D = n))
plt.ylabel('Slope')
plt.xlabel('Dates')
plt.bar(df1.Date, df1['FD Adj'],color = 'r', label = '{D} day Slope Adjusted Close'.format(D = n))
plt.bar(df1.Date, df1['FD MA'], color = 'b', label = '{D} day Slope moving average'.format(D = n))

plt.legend()
plt.show()

plt.title('{D} day Moving slope'.format(D = n))
plt.ylabel('Slope')
plt.xlabel('Dates')
plt.bar(df1.Date, df1['SD Adj'],color = 'r', label = '{D} day Slope Adjusted Close'.format(D = n))
plt.bar(df1.Date, df1['SD MA'], color = 'b', label = '{D} day Slope moving average'.format(D = n))

plt.legend()
plt.show()
'''
