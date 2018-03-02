import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

np.random.seed(100)


flower_data = pd.read_csv('data/Iris_Dataset.csv')
print(flower_data.info())

sepal_l_labels = np.unique(flower_data['sepallength'])
sepal_w_labels = np.unique(flower_data['sepalwidth'])

#describe dataset
flower_data.describe()
#check null values
pd.isnull(flower_data)

figure, axis = plt.subplots(1,2, figsize=(10,5))
sepal_l_colors = np.random.rand(8,4)
sepal_w_colors = np.append(sepal_l_colors,np.random.rand(1,4), axis=0)


X = flower_data.ix[:, 0:4]
flower_data.Class = pd.Categorical(flower_data.Class)
flower_data['cls'] = flower_data.Class.cat.codes
Y = to_categorical(np.ravel(flower_data.cls), num_classes=3)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.33, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(8,activation='relu',input_shape=(4,)))
model.add(Dense(3, activation='softmax'))

model.output_shape
model.summary()
model.get_config()
model.get_weights()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train,Y_train, epochs=100, batch_size=5, verbose=1)
Y_pred = np.rint(model.predict(X_test))
print(str(Y_test))
print(str(Y_pred))

score = model.evaluate(X_test,Y_test)
print(score)




for i in range(len(sepal_l_colors)):
    sepal_l_y = flower_data['sepallength'][flower_data.sepallength == sepal_l_labels[i]]
    sepal_l_x = flower_data['Class'][flower_data.sepallength == sepal_l_labels[i]]
    axis[0].scatter(sepal_l_y, sepal_l_x, c=sepal_l_colors[i])


for i in range(len(sepal_w_colors)):
    sepal_w_y = flower_data['sepalwidth'][flower_data.sepalwidth == sepal_w_labels[i]]
    sepal_w_x = flower_data['Class'][flower_data.sepalwidth == sepal_w_labels[i]]
    axis[1].scatter(sepal_w_y, sepal_w_x, c=sepal_w_colors[i])

#axis.hist(flower_data.sepallength, 10, facecolor='blue', alpha=0.5, label='Sepal Length')
#axis.scatter(flower_data['sepallength'],flower_data['sepalwidth'], color='red')
axis[0].set_ylim([0,8])
axis[0].set_xlim([0,10])
axis[0].set_ylabel('Flower Class')
axis[0].set_xlabel('Sepal Length')

axis[1].set_ylim([0,10])
axis[1].set_xlim([0,19])
axis[1].set_ylabel('Flower Class')
axis[1].set_xlabel('Sepal Width')

axis[0].legend(sepal_l_labels, loc='best')
axis[1].legend(sepal_w_labels, loc='best')

figure.subplots_adjust(left=0.2,right=0.8, bottom=0.5, top=0.8, hspace=0.05, wspace=1)

figure.suptitle('Sepal Length Vs Sepal Width for each Class of Flowers')
plt.show()



