from __future__ import print_function
import tensorflow as tf,numpy as np,math,random
from tensorflow.python import debug as tf_debug
tf.enable_eager_execution()
# tf.keras.backend.set_session(tf_debug.TensorBoardDebugWrapperSession(tf.Session(),'localhost:6064'))
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
#open some data, like the comedically ill-suited Star Wars script
file=open("script","r")
wf=file.read()#the whole script
wordSize=10#characters, which are flattened to "binary" (or 0.0v1.0) 128-length arrays
quoteSize=10#words
numQuotes=10#int(len(wf)/(wordSize*quoteSize))#pretty obvious
flattened=np.zeros(numQuotes*wordSize*quoteSize*128).reshape(numQuotes,wordSize*quoteSize*128)
for quote in range(0,numQuotes):
    iterator=0
    for word in range(0,quoteSize):
        for character in range(0,wordSize):
            for indicator in range(0,128):
                if indicator==ord(wf[quote*quoteSize+word*wordSize+character]):
                    flattened[quote][iterator]=1.0
                iterator+=1
#define a (non-Expanse) spacing function that ensures each datapoint is arbitrarily as far from the other as possible.
def invdistp(y,y_pred):
    lamb=lambda val : val
    total=tf.square(tf.subtract(tf.map_fn(lamb,tf.abs(tf.subtract(y,y_pred))),.1))
    return -((tf.reduce_sum(tf.map_fn(lamb,tf.abs(tf.subtract(y,y_pred))))))
#define an arbitrary classifier
print("Initializing classifier")
numOut=10
classifier=tf.keras.Sequential([
tf.keras.layers.Dense(quoteSize*wordSize*128,activation='relu',input_shape=[quoteSize*wordSize*128]),
tf.keras.layers.Dense(int(.5*quoteSize*wordSize*128),activation='sigmoid'),
tf.keras.layers.Dense(int(.5*quoteSize*wordSize*128),activation='sigmoid'),
tf.keras.layers.Dense(int(.5*quoteSize*wordSize*128),activation='sigmoid'),
tf.keras.layers.Dense(int(numOut),activation='sigmoid')
])
print("Compiling")
classifier.compile(optimizer='sgd',loss=invdistp)
classifier.summary()
print("Fitting")
xs=np.zeros((1,int(quoteSize*wordSize*128)))
classifierStatusQuo=np.zeros((len(flattened),numOut))
numEpochs=10
for epoch in range(0,numEpochs):
    for valin in range(0,len(flattened)):
        xs[0]=flattened[valin]
        ys1=classifier.predict_on_batch(xs)
        if epoch==numEpochs-1:
            classifierStatusQuo[valin]=ys1
        for valout in range(0,(len(flattened)+1)/2):
            if valin!=valout:
                xs2=np.empty((1,int(quoteSize*wordSize*128)))
                xs2[0]=flattened[valout]
                print("Epoch",epoch+1,"/",numEpochs)
                classifier.fit(xs2,ys1,epochs=2,batch_size=1)

print("Initializing generator")
numIn=int(.1*quoteSize*wordSize*128)
generator=tf.keras.Sequential([
tf.keras.layers.Dense(numIn,activation='relu',input_shape=[int(.1*quoteSize*wordSize*128)]),
tf.keras.layers.Dense(int(.3*quoteSize*wordSize*128),activation='sigmoid'),
tf.keras.layers.Dense(int(.5*quoteSize*wordSize*128),activation='sigmoid'),
tf.keras.layers.Dense(int(.8*quoteSize*wordSize*128),activation='sigmoid'),
tf.keras.layers.Dense(quoteSize*wordSize*128,activation='sigmoid')
])
print("Compiling")
generator.compile(optimizer='sgd',loss='mean_squared_error',metrics=['accuracy'])
print("Fitting")
xseeds=np.zeros((len(flattened),1,numIn))
for i in range(0,len(flattened)):
    for j in range(0,numIn):
        xseeds[i][0][j]=random.random()
numGenEpochs=11
expected=np.zeros((1,quoteSize*wordSize*128))
expected[0][0]=-1
for oepoch in range(0,numGenEpochs):
    for epoch in range(0,len(flattened)):
        attempt=classifier.predict_on_batch(generator.predict_on_batch(xseeds[epoch]))
        for point in range(0,len(flattened)):
            cost=tf.reduce_sum(tf.square(tf.subtract(classifierStatusQuo[point],attempt)))
            set=lambda:cost
            ignore=lambda:expected[0][0]
            expected[0][0]=tf.cond(cost<expected[0][0],set,ignore)
            expected[0][0]=tf.cond(expected[0][0]==-1,set,ignore)
        generator.fit(xseeds[epoch],expected,epochs=2,steps_per_epoch=1,verbose=1)
while True:
    xseed=np.zeros((1,numIn))
    for i in range(0,numIn):
        xseed[0][i]=random.random()
    print(generator.predict(xseed))
