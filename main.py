import keras
from keras.layers import Conv2D,BatchNormalization,Dense,AveragePooling2D,Flatten
from keras.models import Sequential,load_model
from keras.utils import to_categorical
from keras.optimizers import Adam
import numpy as np
import os
import random
from skimage.io import imread,imread_collection
from sklearn.svm import SVC
y_sample=[]
x_sample=[]


PATH="D:\\Users\\Otto Glass\\Documents\\MATLAB\\DB\\1"
files=os.listdir(PATH)
for file in files:
    y_sample.append(0)
    x_sample.append(os.path.join(PATH,file))

PATH="D:\\Users\\Otto Glass\\Documents\\MATLAB\\DB\\2"
files=os.listdir(PATH)
for file in files:
    y_sample.append(1)
    x_sample.append(os.path.join(PATH,file))

stuff=list(zip(x_sample,y_sample))
random.shuffle(stuff)
x_random,y_random=zip(*stuff)
x_random=np.array(x_random)
y_random=np.array(y_random)


def Dataset_batcher(X,y,batch_size=200):
    batch_size=len(X)/batch_size
    X_batches=np.array_split(X,batch_size)
    y_batches=np.array_split(y,batch_size)
    while 1:
        for b,X_batch_path in enumerate(X_batches):
            x_batch=np.array(imread_collection(X_batch_path))
            y_batch=to_categorical(y_batches[b])
            yield x_batch , y_batch

model=Sequential([
    Conv2D(80,(1,10),input_shape=(41,125,3),activation='relu'),
    Conv2D(80,(5,1),activation='relu'),
    AveragePooling2D(),
    BatchNormalization(),
    Conv2D(80,(1,5),input_shape=(41,125,3),activation='relu'),
    AveragePooling2D(),
    BatchNormalization(),
    Conv2D(160,(3,3),activation='relu'),
    Conv2D(200,(2,2),activation='relu'),
    BatchNormalization(),
    AveragePooling2D(),
    Flatten(),
    Dense(500,activation='relu'),
    Dense(300,activation='relu'),
    Dense(2,activation='softmax')
])
#adam=Adam(lr=0.0001)
adam=Adam(lr=0.0000001)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(
    Dataset_batcher(x_random,y_random,batch_size=100),
    epochs=400,
    steps_per_epoch=50
    )

#EVAL
y_eval=[]
x_eval=[]


PATH="D:\\Users\\Otto Glass\\Documents\\MATLAB\\DBe\\1"
files=os.listdir(PATH)
for file in files:
    y_eval.append(0)
    x_eval.append(os.path.join(PATH,file))

PATH="D:\\Users\\Otto Glass\\Documents\\MATLAB\\DBe\\2"
files=os.listdir(PATH)
for file in files:
    y_eval.append(1)
    x_eval.append(os.path.join(PATH,file))

x_eval_img=np.array(imread_collection(x_eval))
print("CNN %f"%(sum(model.predict_classes(x_eval_img)==y_eval)/len(y_eval)))

lay=model.layers[:-1]
filtercnn=Sequential(lay)
x_svm=filtercnn.predict(np.array(imread_collection(x_sample)))
lsvm=SVC(kernel='linear')
lsvm.fit(x_svm,y_sample)
gsvm=SVC(kernel='rbf')
gsvm.fit(x_svm,y_sample)
x_svm_eval=filtercnn.predict(x_eval_img)
print("L-SVM %f"%(sum(lsvm.predict(x_svm_eval)==y_eval)/len(y_eval)))
print("G-SVM %f"%(sum(gsvm.predict(x_svm_eval)==y_eval)/len(y_eval)))

modelgan=load_model("saved_model\epochs\discriminator_model18000.h5")
#modelgan.load_weights("saved_model\\ccgan_bci_discriminator.json")
lay=modelgan.layers[0].layers[:-1]
filtergan=Sequential(lay)
x_svm=filtergan.predict(np.array(imread_collection(x_sample)))
lsvm=SVC(kernel='linear')
lsvm.fit(x_svm,y_sample)
gsvm=SVC(kernel='rbf',gamma='auto')
gsvm.fit(x_svm,y_sample)
x_svm_eval=filtergan.predict(x_eval_img)
print("L-SVM %f"%(sum(lsvm.predict(x_svm_eval)==y_eval)/len(y_eval)))
print("G-SVM %f"%(sum(gsvm.predict(x_svm_eval)==y_eval)/len(y_eval)))
