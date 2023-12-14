from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization

import tensorflow as tf

class ResnetBlock(Model):
#out_sized1 = 256, out_sized2 = 256, out_sized3 = 256,
    def __init__(self, out_sized1 = 1024, out_sized2 = 1024, out_sized3 = 1024, dropout_rate = 0.1, dropout_s=False, Layer_Nor=False):
        super(ResnetBlock, self).__init__()
        self.dropout_s = dropout_s
        self.Layer_Nor = Layer_Nor

        self.d1 = Dense(out_sized1, activation='relu', use_bias=False, kernel_initializer='he_uniform')
        self.d2 = Dense(out_sized2, activation='linear', use_bias=False, kernel_initializer='he_uniform')
        self.d3 = Dense(out_sized3, activation='linear', use_bias=False, kernel_initializer='he_uniform')
     
        if dropout_s:
            self.drop = Dropout(dropout_rate)
        
        if Layer_Nor:
            self.ln = LayerNormalization()
            
#         self.a1 = Activation('relu')        

    def call(self, inputs):
        residual = self.d3(inputs)  # residual等于输入值本身，即residual=x
 
        x = self.d1(inputs)
        x = self.d2(x)     
        if self.dropout_s:
            x = self.drop(x)
            
        if self.Layer_Nor:
            out = self.ln(x+residual)
        else:
            out = x+residual
        
        return out
    
class TiED(Model):

    def __init__(self, en_lyers, de_lyers, med_size = 1024,outputsize = 1):  # block_list表示每个block有几个卷积层
        super(TiED, self).__init__()

        self.encoders = tf.keras.models.Sequential()
        self.decoders = tf.keras.models.Sequential()
        
        for i in range(en_lyers):  
            blocke = ResnetBlock(out_sized1 = med_size, out_sized2 = med_size, out_sized3 = med_size, dropout_s=True, Layer_Nor=True)
            self.encoders.add(blocke)  # 将构建好的block加入resnet
 
        for i in range(de_lyers):  
            blockd = ResnetBlock(out_sized1 = med_size, out_sized2 = med_size, out_sized3 = med_size, dropout_s=True, Layer_Nor=True)
            self.decoders.add(blockd)  # 将构建好的block加入resnet

        #输出resnetblock层
        self.res1 = ResnetBlock(out_sized1 = med_size, out_sized2 = outputsize, out_sized3 = outputsize, dropout_s=True, Layer_Nor=True)
        
        self.d1 = Dense(outputsize, activation='linear', use_bias=False, kernel_initializer='he_uniform')
        
    def call(self, inputs):
        residual = self.d1(inputs)
        
        x = self.encoders(inputs)
        x = self.decoders(x)
        x = self.res1(x)
        
        y = x + residual
        return y