# 다 : 다
#실습
# 1. R2             : 0.5 이하 / 음수 안됨
# 2. layer          : 5개 이상
# 3. node           : 각 10개 이상
# 4. batch_size     : 8 이하
# 5. epochs         : 30 이상


import numpy as np

#1. data
x = np.array([range(100), range(301,401), range(1,101), range(200,300), range(401,501)])
y = np.array([range(711,811), range(201,301), ])

x_pred2 = np.array([100,401,101,300,501])

print(x.shape)          #(3,100)
print(y.shape)          #(3,100)
 
x = np.transpose(x)
y = np.transpose(y)

x_pred2 = x_pred2.reshape(1,5)

print(x.shape)          #(100,3)
print(y.shape)          #(100,3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,train_size = 0.8, test_size=0.2, shuffle=True, random_state=66 )
print(x_train.shape)    #(80,5)
print(y_train.shape)    #(80,2)

#2. model config

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# from keras.layers import Dense

model=Sequential()
model.add(Dense(3,input_dim=5))
model.add(Dense(1))
model.add(Dense(2))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(2))

#3. Compilem,run
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train,y_train,epochs=110,batch_size=5,validation_split=0.2,verbose=0)

#4 Evaluation validation
loss, mae = model.evaluate(x_train,y_train)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x_test)
print("y_predict : ",  y_predict)
print("y_predict : ",  y_predict[10:11])

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE :", RMSE(y_test,y_predict))
# print("mse : ", mean_squared_error(x_test, y_test))
print("mse : ", mean_squared_error(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 :",r2 )

# y_Pred2 = model.predict(x_pred2)

    # print("y_pred2 : " , y_Pred2)
    # [[811.00024 301.0001 ]]

"""
model=Sequential()
model.add(Dense(1,input_dim=5))
model.add(Dense(5))
model.add(Dense(1))
model.add(Dense(5))
model.add(Dense(1))
model.add(Dense(2))

#3. Compilem,run
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train,y_train,epochs=80,batch_size=5,validation_split=0.2)

     R2 : 0.05644075330757675
     R2 : -25.52750751716041
     R2 : -23.496960584086516
     R2 : -19.847628511979174
     R2 : -0.83965069527159


model=Sequential()
model.add(Dense(1,input_dim=5))
model.add(Dense(5))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(5))
model.add(Dense(2))

#3. Compilem,run
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train,y_train,epochs=80,batch_size=5,validation_split=0.2)

    R2 : -2.498732733410298
    R2 : -4.5062529793354225
    R2 : -1.6792232925190151
    R2 : -0.2868367032197777
    R2 : -1.1257462530901758

model=Sequential()
model.add(Dense(1,input_dim=5))
model.add(Dense(5))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(6))
model.add(Dense(2))

#3. Compilem,run
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train,y_train,epochs=80,batch_size=5,validation_split=0.2)

    R2 : 0.7759595215373379
    R2 : 0.6208624712062935
    R2 : -0.7764782299080455
    R2 : 0.7628163165956392
    R2 : -0.8019591031019231

model=Sequential()
model.add(Dense(1,input_dim=5))
model.add(Dense(5))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(6))
model.add(Dense(2))

#3. Compilem,run
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train,y_train,epochs=90,batch_size=5,validation_split=0.2)

    R2 : 0.493570149920223
    R2 : 0.7782239092606429
    R2 : -0.4635936547842817
    R2 : 0.5173760373310362
    R2 : 0.4069597149091605


model=Sequential()
model.add(Dense(1,input_dim=5))
model.add(Dense(5))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(6))
model.add(Dense(2))

#3. Compilem,run
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train,y_train,epochs=1000,batch_size=5,validation_split=0.2,verbose=0)

    R2 : -0.778881700649019
    R2 : -1.0573433571754556
    R2 : 0.46241985189169843
    R2 : 0.6255223050282073
    R2 : 0.7814088036008122

model=Sequential()
model.add(Dense(1,input_dim=5))
model.add(Dense(5))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(7))
model.add(Dense(2))

#3. Compilem,run
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train,y_train,epochs=100,batch_size=5,validation_split=0.2,verbose=0)

    R2 : 0.5315185155676185
    R2 : 0.7839974040284368
    R2 : 0.14610635507973746
    R2 : 0.14328969644737266
    R2 : 0.7189283343284415


model=Sequential()
model.add(Dense(4,input_dim=5))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(2))

#3. Compilem,run
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train,y_train,epochs=110,batch_size=5,validation_split=0.2,verbose=0)

    R2 : 0.4842424657775097
    R2 : 0.16332580732371677
    R2 : 0.18592896618195814
    R2 : 0.7614725681459917
    R2 : 0.5487422985256822
    R2 : 0.6108107764112538
    R2 : 0.7666133507586737
    R2 : -5.3758375042366096
    R2 : 0.27610302962509375
    R2 : 0.6815244387771704

model=Sequential()
model.add(Dense(3,input_dim=5))
model.add(Dense(1))
model.add(Dense(3))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(2))

#3. Compilem,run
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train,y_train,epochs=110,batch_size=5,validation_split=0.2,verbose=0)

    R2 : 0.433416683599063
    R2 : 0.5487172104583875
    R2 : 0.18498126212958138
    R2 : 0.6415213455904707
    R2 : 0.7821562160922273
    R2 : 0.48197430433101385
    R2 : 0.5688086420398191
    R2 : 0.7572807954496465
    R2 : 0.2986814557806095
    R2 : 0.769721394683122
"""
