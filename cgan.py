'''
C-DCGAN on MNIST using Keras
'''

import numpy as np
import time
#from tensorflow.examples.tutorials.mnist import input_data
from keras.utils import to_categorical
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Activation, Flatten, Reshape,Cropping2D
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D,AveragePooling2D,Concatenate
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

from skimage.io import imread_collection

import os
import matplotlib.pyplot as plt

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class CGAN(object):
    def __init__(self, img_rows=41, img_cols=125, channel=3):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D
        #filter
        self.filter_cnn=Sequential([
            Conv2D(80,(1,10),input_shape=(41,125,3),activation='relu'),
            Conv2D(80,(5,1),activation='relu'),
            AveragePooling2D(),
            BatchNormalization(),
            Conv2D(80,(1,5),activation='relu'),
            AveragePooling2D(),
            BatchNormalization(),
            Conv2D(160,(3,3),activation='relu'),
            Conv2D(200,(2,2),activation='relu'),
            BatchNormalization(),
            AveragePooling2D(),
            Flatten(),
            Dense(500,activation='relu'),
            Dense(300,activation='relu'),
        ])
        #discriminator
        img=Input(batch_shape=(None,41,125,3))
        label=Input(batch_shape=(None,2))
        features=self.filter_cnn(img)
        output=Concatenate()([features,label])
        validator=Sequential([
            Dense(10,activation='relu',input_shape=(302,)),
            Dense(1,activation='softmax')
        ])
        valid=validator(output)
        self.D=Model(inputs=[img,label],outputs=valid)

#        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 64+64+64+64
        #dim = 8
        # In: 100
        # Out: dim x dim x depth
        self.generator_model=Sequential([
            Dense(32*11*depth, input_dim=102),
            BatchNormalization(momentum=0.9),
            Activation('relu'),
            Reshape((11, 32, depth)),
            Dropout(dropout),
            UpSampling2D(),
            Conv2DTranspose(int(depth/2), 5, padding='same'),
            BatchNormalization(momentum=0.9),
            Activation('relu'),
            UpSampling2D(),
            Conv2DTranspose(int(depth/4), 5, padding='same'),
            BatchNormalization(momentum=0.9),
            Activation('relu'),
            Conv2DTranspose(int(depth/8), 5, padding='same'),
            BatchNormalization(momentum=0.9),
            Activation('relu'),
            Conv2DTranspose(3, 5, padding='same',activation='softmax'),
            Cropping2D(cropping=((3,0),(3,0)))
        ])
        noise=Input(batch_shape=(None,100))
        label=Input(batch_shape=(None,2))
        input_data=Concatenate()([noise,label])

        image=self.generator_model(input_data)
        self.G=Model(inputs=[noise,label],outputs=[image])
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = Adam(lr=0.01, decay=6e-8)
        self.DM=self.D
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])

        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = Adam(lr=0.05, decay=3e-8)
        noise=Input(batch_shape=(None,100))
        label=Input(batch_shape=(None,2))
        img_fake=self.G([noise,label])
        valid=self.DM([img_fake,label])
        self.AM = Model(inputs=[noise,label],outputs=valid)
        self.AM.summary()
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])
        return self.AM

class BCI_CGAN(object):
    def __init__(self):
        self.img_rows = 125
        self.img_cols = 41
        self.channel = 3

        y_sample=[]
        x_sample=[]
        PATH="D:\\Users\\Otto Glass\\Documents\\MATLAB\\DB\\1"
        files=os.listdir(PATH)
        self.x_train=[]
        self.y_train=[]
        for file in files:
            y_sample.append(0)
            x_sample.append(os.path.join(PATH,file))
        y_train=to_categorical(np.array(y_sample),num_classes=2)
        self.x_train.append(np.array(imread_collection(x_sample))/255)
        self.y_train.append(y_train)

        y_sample=[]
        x_sample=[]
        PATH="D:\\Users\\Otto Glass\\Documents\\MATLAB\\DB\\2"
        files=os.listdir(PATH)
        for file in files:
            y_sample.append(1)
            x_sample.append(os.path.join(PATH,file))
        y_train=to_categorical(np.array(y_sample),num_classes=2)
        self.x_train.append(np.array(imread_collection(x_sample))/255)
        self.y_train.append(y_train)

        self.CGAN = CGAN()
        self.CGAN.discriminator()
        self.discriminator =  self.CGAN.discriminator_model()
        self.generator = self.CGAN.generator()
        self.adversarial = self.CGAN.adversarial_model()

    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        for i in range(train_steps):

            #class1
            #discriminator
            indices=np.random.randint(0,self.x_train[0].shape[0], size=batch_size)
            images_train = self.x_train[0][indices, :, :, :]
            images_label = self.y_train[0][indices, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            input_data=np.concatenate((images_label[:,0:2],noise),1)
            images_fake = self.generator.predict([noise,images_label])
            x = np.concatenate((images_train, images_fake))
            x_1=np.concatenate((images_label, images_label))
            #y = np.ones([2*batch_size, 1])
            y = np.zeros([2*batch_size, 1])
            #y[batch_size:, :] = 0
            y[batch_size:, :] = 1
            d_loss = self.discriminator.train_on_batch([x,x_1], y)
            #generator
            #y = np.ones([batch_size, 1])
            y = np.zeros([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch([noise,images_label], y)

            #class2
            #discriminator
            indices=np.random.randint(0,self.x_train[1].shape[0], size=batch_size)
            images_train = self.x_train[1][indices, :, :, :]
            images_label = self.y_train[1][indices, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            input_data=np.concatenate((images_label,noise),1)
            images_fake = self.generator.predict(input_data)
#            label_fake=to_categorical(np.ones((batch_size))*2)
            x = np.concatenate((images_train, images_fake))
            x_1 = np.concatenate((images_label,images_label))
            #y = np.ones([2*batch_size, 1])
            y = np.zeros([2*batch_size, 1])
            #y[batch_size:, :] = 0
            y[batch_size:, :] = 1
            d_loss += self.discriminator.train_on_batch([x,x_1], y)
            #generator
            #y = np.ones([batch_size, 1])
            y = np.zeros([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss += self.adversarial.train_on_batch([noise,images_label], y)

            #report
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0]+d_loss[2], (d_loss[1]+d_loss[3])/2)
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0]+a_loss[2], (a_loss[1]+a_loss[3])/2)
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, step=(i+1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        label = to_categorical(np.random.randint(0,2,samples))
        noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
        input_data=np.concatenate((label,noise),1)
        filename = "images\\%d.png" % step
        images = self.generator.predict(input_data)

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols,self.channel])
            plt.imshow(image)
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
            self.save("model.h5")
        else:
            plt.show()
    def save(self,filename):
        self.CGAN.D.save("saved_model\\filter_"+filename)
        self.CGAN.D.save_weights("saved_model\\filter_bci.json")
        self.discriminator.save("saved_model\\discriminator_"+filename)
        self.discriminator.save_weights("saved_model\\discriminator_bci.json")
        self.generator.save("saved_model\\generator_"+filename)
        self.generator.save_weights("saved_model\\generator_bci.json")

if __name__ == '__main__':
    ccgan = BCI_CGAN()
    ccgan.train(train_steps=80000, batch_size=50, save_interval=200)