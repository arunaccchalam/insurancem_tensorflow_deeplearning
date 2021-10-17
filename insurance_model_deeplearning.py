import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# reading the csv file 
insurance = pd.read_csv("insurance.csv")
print(insurance.head())
print(len(insurance))

#splitting x and y fro the original data set
x = insurance.drop('charges',axis=1)
y = insurance['charges']
print(x)
print(y)

# splitting into train and test set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
len(x_train),len(x_test),len(y_train),len(y_test)
x_test.columns

# converting the strings into tensors using scaling
# creating column transformer
ct = make_column_transformer(
     (MinMaxScaler(),['age','bmi','children']), # turn all values in this column between 0 and 1
     (OneHotEncoder(handle_unknown='ignore'),['sex','smoker','region']))

# fitting the column transformer on x_train
ct.fit(x_train)
x_train_ct = ct.transform(x_train)
x_test_ct = ct.transform(x_test)


#visualising our scaled and OneHotEncoded data
print(x_train_ct[0])
print(x_train.loc[0])
print(x_train.shape,x_train_ct.shape)
print(tf.random.set_seed(42))

# creating the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Dense(1,activation='relu')
    
])

# compiling the model
model.compile(loss=tf.keras.losses.mae,
             optimizer=tf.keras.optimizers.Adam(lr=0.01),
             metrics=['mae'])

history = model.fit(x_train_ct,y_train,epochs=200)

#plotting loss curves
pd.DataFrame(history.history).plot()
plt.xlabel('epochs')
plt.ylabel('loss')

print(y.mean())

#evaluating and making predictions 
model.evaluate(x_test_ct,y_test)
preds = model.predict(x_test_ct)

#calculating the model results
def results(y_pred,y_true):
    a = mean_absolute_error(y_pred,y_true)
    b = mean_squared_error(y_pred,y_true)
    c = np.sqrt(b)
    results = {
        'MEAN ABSOLUTE ERROR' : a,
        'MEAN SQUARED ERROR'  :b ,
        'MEAN ROOTED ERROR'   :c}
    return results;
model_results = results(preds,y_test)
print(model_results)
model.save('insurance_model_dl/HDF5')