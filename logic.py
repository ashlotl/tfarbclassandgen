from __future__ import print_function
import tensorflow as tf,numpy as np,math,pygame
pygame.init()
screen=pygame.display.set_mode((500,500))
pygame.display.set_caption("Scary port v3084")
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
#open some data, like the comedically ill-suited Star Wars script
file=open("script","r")
wf=file.read()#the whole script
wordSize=10#characters, which are flattened to "binary" (or 0.0v1.0) 128-length arrays
quoteSize=10#words
numQuotes=3#int(len(wf)/(wordSize*quoteSize))#pretty obvious
flattened=np.zeros(numQuotes*wordSize*quoteSize*128).reshape(numQuotes,wordSize*quoteSize*128)
for quote in range(0,numQuotes):
    iterator=0
    # print(quote)
    for word in range(0,quoteSize):
        for character in range(0,wordSize):
            for indicator in range(0,128):
                if indicator==ord(wf[quote*quoteSize+word*wordSize+character]):
                    flattened[quote][iterator]=1.0
                iterator+=1
print(len(flattened))
total=0
epochid=0
def fact(num):
    if num!=1:
        return num*fact(num-1)
#define a (non-Expanse) spacing function that ensures each datapoint is arbitrarily as far from the other as possible.
def check((ye,ype)):
    if ye!=ype:
        return (1, 1)
    return (0,0)

def invdistp(y,y_pred):
    # print(y,y_pred)
    # total=tf.Variable(initial_value=tf.constant(.1),trainable=False)
    if tf.map_fn(check,(y,y_pred))==(0,0):
        return 0
    # bad=True
    # i=0
    #
    # for e in y:
    #     if y[i]!=y_pred[i]:
    #         bad=False
    #         break
    #     i+=1
    # if bad:
    #     return 0
    print("SHAPE:",y.shape,y_pred.shape)
    lamb=lambda val : val
    total=tf.map_fn(lamb,tf.square(tf.subtract(y,y_pred)))*1000
    print("HERE:",total)
    return tf.math.reciprocal(tf.reduce_sum(total))
# invdist=tf.contrib.eager.function(invdistp)
#define an arbitrary classifier
print("Initializing classifier")
classifier=tf.keras.Sequential([
tf.keras.layers.Dense(quoteSize*wordSize*128,activation='relu',input_shape=[quoteSize*wordSize*128]),
tf.keras.layers.Dense(int(.5*quoteSize*wordSize*128),activation='sigmoid'),
tf.keras.layers.Dense(int(2),activation='sigmoid')
])
# classifier.build()
# predictor=classifier
print("Compiling")
classifier.compile(optimizer='sgd',loss=invdistp)
classifier.summary()
print("Fitting")
z=True
for epoch in range(0,50):
    screen.fill((255,255,255))
    for valin in range(0,len(flattened)):
        for valout in range(0,len(flattened)):
            if valin!=valout:
                xs=np.zeros((1,int(quoteSize*wordSize*128)))
                print(flattened.shape)
                xs[0]=flattened[valin]
                print(xs)
                # predictor.set_weights(classifier.get_weights())
                # if z:
                #     ys=np.ones((1,int(.1*quoteSize*wordSize*128)))
                # else:
                #     ys=np.zeros((1,int(.1*quoteSize*wordSize*128)))
                # z=not z
                ys1=classifier.predict_on_batch(xs)
                # classifier.trainable=True
                xs2=np.empty((1,int(quoteSize*wordSize*128)))
                xs2[0]=flattened[valout]
                ys2=classifier.predict_on_batch(xs2)
                ys=np.zeros((1,int(2)))
                total=0
                color=(0,0,0)
                pygame.draw.circle(screen,color,(int(200*(ys2[0][0])),int(200*(ys2[0][1]))),2,0)
                for index in range(0,len(ys2)):
                    total+=(ys2[0][index]-ys1[0][index])**2
                total=-(total)
                if total==0:
                    total=.0000001
                for index in range(0,len(ys2[0])):
                    ys[0][index]=ys2[0][index]+.00001*(ys2[0][index]-ys1[0][index])/total
                    if ys[0][index]>1:
                        ys[0][index]=1
                    if ys[0][index]<0:
                        ys[0][index]=0
                print("fit...",xs2.shape,ys.shape)
                # tf.reset_default_graph()
                # classifier.compile(optimizer='sgd',loss=invdist,metrics=['accuracy'])
                classifier.fit(xs2,ys1,epochs=1,batch_size=1)
    pygame.display.flip()
#custom loss that allows one prediction to be compared to many expecteds with a 'nearest' behaviour
def lossf(y,y_pred):
    return 1-y_pred
#the input to generator could be different and is not correlated to the output from classifier, we simply do this for some symmetry
print("Initializing generator")
generator=tf.keras.Sequential([
tf.keras.layers.Dense(int(.1*quoteSize*wordSize*128),activation='relu',input_shape=[int(.1*quoteSize*wordSize*128)]),
tf.keras.layers.Dense(int(.3*quoteSize*wordSize*128),activation='sigmoid'),
tf.keras.layers.Dense(int(.5*quoteSize*wordSize*128),activation='sigmoid'),
tf.keras.layers.Dense(int(.8*quoteSize*wordSize*128),activation='sigmoid'),
tf.keras.layers.Dense(quoteSize*wordSize*128,activation='sigmoid')
])
print("Compiling")
generator.compile(optimizer='adam',loss=lossf,metrics=['accuracy'])
print("Fitting")
generator.fit(tf.ones([1,200]),tf.zeros([1,200]),epochs=10,steps_per_epoch=1,verbose=1)
print(generator.predict(tf.ones([1,200]),steps=1))
