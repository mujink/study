import numpy as np

f = lambda x: x**2 - 4*x+6
# 인풋 차원이 x,y 2인 2개의 노드를 가진 레이어의 2차 다항식 표현
gradient = lambda x : 2*x -4
# 가중치 2, 바이어스 -4 => 모든 레이어의 다항식의 미분값이 그라디언트가 됨

x0 = -5.0 
epoch = 80
learning_rate = 0.1

print("step\tx\tf(x)")
print("{:02d}\t{:6.5f}\t{:6.5f}".format(0, x0, f(x0)))

for i in range (epoch):
    temp = x0 - learning_rate * gradient(x0)
    # 원값 - 러닝레이트 * 인풋레이어 노드 미분값(그라디언트)
    x0 = temp
    print("{:02d}\t{:6.5f}\t{:6.5f}".format(i+1, x0, f(x0)))
