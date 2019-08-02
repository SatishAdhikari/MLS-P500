from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
#from FisrtDerivative import training_points, training_labels
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

training_points = pd.read_csv('training_points.csv').drop(columns = ['0'])
training_labels = pd.read_csv('training_labels.csv').drop(columns = ['0'])

print(training_points, training_labels)
scaler = MinMaxScaler()
training_points_scaled = scaler.fit_transform(training_points)

#from Prediction import prediction
X_train, X_test, y_train, y_test = train_test_split(training_points_scaled, training_labels, test_size=0.20, random_state=15) #42

'''
cv_scores = []
length = range(50,100)
for x in length:
    classifier = KNeighborsClassifier(n_neighbors= x)
    classifier.fit(X_train, y_train)
    scores = classifier.score(X_test, y_test)
    cv_scores.append(scores)

plt.title('Neighbors versus accuracy')
plt.xlabel('K Number of neighbors')
plt.ylabel('Percentage Accuracy')
plt.plot(length, cv_scores, color = 'r')
plt.show()
'''
classifier = KNeighborsClassifier(n_neighbors=97)
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print('Precent accuracy of the K-Nearest Neighbors algo', round(score* 100, 2), '%')

#Prediction_probability = classifier.predict_proba(prediction)
#Prediction = classifier.predict(prediction)

#print(Prediction_probability, Prediction)
# These dates may not be accurate the last 1 should be refrencing tomorrow while the index 214 is today
