# [실습]
# 덧셈
# 뺄셈
# 곱셈
# 나눗셈
# 맹그러라!!!

import tensorflow as tf
node1 = tf.constant(2.0)
node2 = tf.constant(3.0)
# node1 = tf.constant(2.0,3.0,4.0)      # Expected uint8, got 3.0 of type 'float' instead
# node2 = tf.constant(3.0,4.0,5.0)      # Expected uint8, got 3.0 of type 'float' instead

node3 = tf.add(node1, node2)
node4 = tf.subtract(node1, node2)
node5 = tf.multiply(node1, node2)
node6 = tf.div(node1, node2)
node7 = tf.mod(node1, node2)

# sess = tf.compat.v1.Session()
sess = tf.Session()
print(sess.run(node3))
print(sess.run(node4))
print(sess.run(node5))
print(sess.run(node6))
print(sess.run(node7))


# 5.0
# -1.0
# 6.0
# 0.6666667
# 2.0

# ======================== 일반 연산 ====================
# tf.add                    # 덧셈 
# tf.sub                    # 뺄셈
# tf.mul                    # 곱셈 
# tf.div                    # 나눗셈의 몫 
# tf.mod                    # 나눗셈의 나머지 
# tf.abs                    # 절댓값을 리턴 
# tf.neg                    # 음수를 리턴
# tf.sign                   # 부호를 리턴 (음수는 -1, 양수는 1, 0은 0)
# tf.inv                    # 역수를 리턴 (예: 3의 역수는 1/3)
# tf.squae                  # 제곱을 계산 
# tf.round                  # 반올림 값을 리턴 
# tf.sqrt                   # 제곱근을 계산 
# tf.pow                    # 거듭제곱 값을 계산 
# tf.exp                    # 지수 값을 계산 
# tf.log                    # 로그 값을 계산 
# tf.maximum                # 최댓값을 리턴  
# tf.minimum                # 최솟값을 리턴 
# tf.cos                    # 코사인 함수 값을 계산 
# tf.sin                    # 사인 함수 값을 계산 
# ======================== 텐서 연산 ====================
# tf.diag                   #  대각행렬을 리턴
# tf.transpose             #  전치행렬을 리턴
# tf.matmul                #  두 텐서를 행렬곱한 결과 텐서를 리턴
# tf.matrix_determinant    #  정방행렬의 행렬식 값을 리턴
# tf.matrix_inverse        #  정방행렬의 역행렬을 리턴


# 출처: https://eehoeskrap.tistory.com/188 [Enough is not enough]