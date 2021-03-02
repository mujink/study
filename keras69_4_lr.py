# 이 소스의 목적은 y_predict 0.8을 찾아내는 것!!
x_train = 0.5
y_train = 0.8

# ===============================================
# 다차원 입력에 대한 미분값이 웨이트가 됨
weight = 0.5
lr = 0.01
# 반복 횟수
epoch = 150
# ===============================================


# 로스와 옵티마이저 동작
# 최적의 로스를 찾는 방법
# x, y 2차원 다항식에서의 예시
for iteration in range(epoch):
    y_predict = x_train*weight
    # MSE Loss
    error = (y_predict - y_train) ** 2
    print("Error : " + str(error) + "\ty_predict :" + str(y_predict))
    # 그레디언트의 방향을 알려줌 
    up_y_predict = x_train * (weight + lr)
    # up_mse 로스
    up_error = (y_train -  up_y_predict) ** 2

    # 그레디언트의 방향을 알려줌
    down_y_predict = x_train * (weight - lr)
    # down_mse 로스
    down_error = (y_train - down_y_predict) **2

    # 최적의 웨이트를 다음과 같이 찾음
    # 로스 값이 최소점이 되는 웨이트
    if (down_error <= up_error):
        weight = weight - lr
    if (down_error > up_error):
        weight = weight + lr

print("weight :", weight)