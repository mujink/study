import tensorflow as tf

tf.compat.v1.set_random_seed(66)

x_train = [1,2,3]
y_train = [3,5,7]

W = tf.Variable(tf.random.normal([1]), name='weight')
B = tf.Variable(tf.random.normal([1]), name='bias')

# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())
# print(sess.run(W), sess.run(b))
hypothesis = x_train * W + B

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
import pandas as pd
steps = []
costs = []
w = []
b = []
dcost = []
prd = []
dw = []
db = []
print(sess.run(W)[0], sess.run(B)[0])
for step in range(80):
    sess.run(train)
    steps.append(step)
    a = sess.run(cost)
    c = sess.run(W)[0]
    e =sess.run(B)[0]
    costs.append(a)
    w.append(c)
    b.append(e)
    dcost.append(sess.run(tf.sqrt(a*3)+y_train))
    prd.append(10*c+e)
    if step==0:
        dw.append(0)
        db.append(0)
    else:
        dw.append(w[-2]-c)
        db.append(b[-2]-e)
    if step % 20 ==0:
        print(step, sess.run(cost), sess.run(W), sess.run(B))

df = pd.DataFrame({"step":steps, "cost":costs, "weight":w, "bias":b, "dcost":dcost, "prd":prd, "dweight":dw, "dbias":db})
df.to_csv("tf05_Linear.csv")

df = pd.read_csv("tf05_Linear.csv", index_col=0 )
del df["step"]
print(df)
import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))
plt.subplot(2,1,1)
plt.plot(steps, df["cost"], color='red')
plt.plot(steps, df["weight"], color='green')
plt.plot(steps, df["bias"], color='purple')
plt.subplot(2,1,2)
plt.scatter(x_train, y_train)
# plt.plot(x_train, ad*df["weight"]+df['bias'], color='blue')
plt.show()