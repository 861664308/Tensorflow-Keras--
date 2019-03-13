import tensorflow as tf
import numpy as np
#计算Legendre多项式及其导数
def Legendre(x, m):
    L = np.zeros(np.max([m, 3]))
    L[0] = 1; L[1] = x; L[2] = 1 / 2 * (3 * x ** 2 - 1)
    if m > 3:
        for i in range(3, m):
            L[i] = ((2 * i - 1) * x * L[i - 1] - (i - 1) * L[i - 2]) / i
#    L = L[-1]
    return L
def Legendredx(x, m):
    L = np.zeros(np.max([m, 3]))
    L1 = Legendre(x, m)
    L[0] = 0; L[1] = 1; L[2] = 3 / 2 * x
    if m > 3:
        for i in range(3, m):
            L[i] = (2 * i - 1) / i * (L1[i - 1] + x * L[i - 1]) - (i - 1) / i * L[i - 2]
#    L = L[-1]
    return L
def Legendred2x(x, m):
    L = np.zeros(np.max([m, 3]))
    L1 = Legendredx(x, m)
    L[0] = 0; L[1] = 0; L[2] = 3 / 2
    if m > 3:
        for i in range(3, m):
            L[i] = (2 * i - 1) / i * (2 * L1[i - 1] + x * L[i - 2]) - (i - 1) / i * L1[i - 2]
#    L = L[-1]
    return L
#计算勒让德多项式及其导数
def computeLegendrefunctionandderivation(X, m):
    L = len(X)
    L1 = []
    L2 = []
    L3 = []
    for i in range(L):
        temp = X[i]
        L1.append(Legendre(temp, m))
        L2.append(Legendredx(temp, m))
        L3.append(Legendred2x(temp, m))
    L1 = np.array(L1)
    L2 = np.array(L2)
    L3 = np.array(L3)
    value_dict = {'function': L1, 'firstderivation': L2, 'secondderivation': L3}
    return value_dict
#计算N(x,p)的导数
def firstdx_N(w, x, dx):
    #L = len(x)
    z = tf.reduce_sum(tf.multiply(tf.cast(w, tf.float64), tf.cast(x, tf.float64)))
    N = tf.nn.tanh(z)
    firstdx_N = (1 - tf.cast(N,tf.float64) ** 2) * tf.reduce_sum(tf.multiply(tf.cast(w,tf.float64), tf.cast(dx,tf.float64)))
    return firstdx_N
def seconddx_N(w, x, dx, dx2):
    z = tf.reduce_sum(tf.multiply(tf.cast(w, tf.float64), tf.cast(x, tf.float64)))
    N = tf.nn.tanh(z)
    seconddx_N = (1 - N ** 2) * tf.reduce_sum(tf.multiply(tf.cast(w, tf.float64), tf.cast(dx2,tf.float64)))\
                 - 2 * N * tf.reduce_sum(tf.multiply(tf.cast(w,tf.float64), tf.cast(dx, tf.float64)))
    return seconddx_N
#计算y=x**2*N(x,p)对x的导数
def dx(w, x, dx1):
    z = tf.reduce_sum(tf.multiply(tf.cast(w, tf.float64), tf.cast(x, tf.float64)))
    N = tf.nn.tanh(z)
    N1 = firstdx_N(w, x, dx1)
    dydx = 2 * x * N + N1 * (x ** 2)
    return dydx
def d2ydx2(w, x, dx, dx2):
    z = tf.reduce_sum(tf.multiply(tf.cast(w, tf.float64), tf.cast(x, tf.float64)))
    N = tf.nn.tanh(z)
    N1 = firstdx_N(w, x, dx)
    N2 = seconddx_N(w, x, dx, dx2)
    d2ydx2 = 2 * N + 4 * x * N1 + N2 * (x ** 2)
    return  d2ydx2
#计算代价函数
def costfunction(w, x, x0 , dx1, dx2):
    z = tf.reduce_sum(tf.multiply(tf.cast(w, tf.float64), tf.cast(x, tf.float64)))
    N = tf.nn.tanh(z)
    dy = dx(w, x, dx1)
    ddy2 = d2ydx2(w, x, dx1, dx2)
    costfunction = ddy2 + 2 / x0 * dy + 8 * tf.exp(x0 ** 2 * N) + 4 * tf.exp(x0 ** 2 * N / 2)
    return costfunction
def loss_function(X, w, Legendrefunction, firstdx, seconddx, length):
    L = []
    for i in range(length):
        temp0 = X[i]
        temp1 = Legendrefunction[i]
        temp2 = firstdx[i,:]
        temp3 = seconddx[i,:]
        temp = costfunction(w, temp1, temp0 , temp2, temp3)
        L.append(temp)
    L = np.array(L)
    L = tf.reduce_sum(tf.multiply(L, L)) / 2
    return L
x = np.linspace(0.1, 1, 10)
n = 5
m = tf.constant(n,'int8')
value_dict = computeLegendrefunctionandderivation(x, n)
Legendrefunction = value_dict['function']
firstdx = tf.Variable(value_dict['firstderivation'])
seconddx = tf.Variable(value_dict['secondderivation'])
#建立输入层
x_train = tf.placeholder('float64', [None, n])
#建立输出层
W = tf.Variable(tf.truncated_normal([n, 1],stddev = 0.1), 'float64')
y_predict = tf.nn.tanh(tf.matmul(tf.cast(x_train, tf.float64), tf.cast(W, tf.float64)))
loss = loss_function(X = tf.Variable(x,'float64'), w = W, Legendrefunction = x_train, firstdx = firstdx, seconddx = seconddx, length=n)
optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.0001).minimize(loss)
Epoch = 10
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(Epoch):
        sess.run(optimizer, feed_dict={x_train: functionvalue})
        #W = tf.Variable(tf.truncated_normal([n, 1], stddev=0.1), 'float')
        #loss = loss_function(X=tf.Variable(x, 'float'), m=m, w=W)
        #optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.0001).minimize(loss)
        print(sess.run(loss))
'''
y_predict = layer(output_dim = 1, input_dim = m, inputs = x_train, activation = tf.nn.tanh)
'''




