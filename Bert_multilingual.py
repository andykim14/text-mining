#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install transformers')


# In[ ]:


import tensorflow as tf
import numpy as np
import pandas as pd
from transformers import *
import json
from tqdm import tqdm
import os


# In[ ]:


import os
from google.colab import drive
drive.mount('/content/gdrive/')


# In[ ]:


os.chdir("/content/gdrive/MyDrive/Colab Notebooks/News_detection/train_test")
os.getcwd()
os.listdir()


# In[ ]:


train = pd.read_csv("train_set.csv")
test = pd.read_csv("test_set.csv")


# In[ ]:


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


# In[ ]:


train['Review'] = (train['title'].map(str) +' '+ train['content']).apply(lambda row: row.strip())
test['Review'] = (test['title'].map(str) +' '+ test['content']).apply(lambda row: row.strip())


# In[ ]:


train[50:70]


# In[ ]:


def convert_data(data_df):
    global tokenizer
    
    SEQ_LEN = 128 
    
    tokens, masks, segments, targets = [], [], [], []
    
    for i in tqdm(range(len(data_df))):
        token = tokenizer.encode(data_df[DATA_COLUMN][i], max_length=SEQ_LEN, truncation=True, padding='max_length')
       
        num_zeros = token.count(0)
        mask = [1]*(SEQ_LEN-num_zeros) + [0]*num_zeros
        
        segment = [0]*SEQ_LEN

        tokens.append(token)
        masks.append(mask)
        segments.append(segment)
        
        targets.append(data_df[LABEL_COLUMN][i])

    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)
    targets = np.array(targets)

    return [tokens, masks, segments], targets

def load_data(pandas_dataframe):
    data_df = pandas_dataframe
    data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
    data_df[LABEL_COLUMN] = data_df[LABEL_COLUMN].astype(int)
    data_x, data_y = convert_data(data_df)
    return data_x, data_y

SEQ_LEN = 128
BATCH_SIZE = 20
DATA_COLUMN = "Review"
LABEL_COLUMN = "information"

train_x, train_y = load_data(train)


# In[ ]:


test_x, test_y = load_data(test)


# In[ ]:


model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
token_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_word_ids')
mask_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_masks')
segment_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_segment')
bert_outputs = model([token_inputs, mask_inputs, segment_inputs])


# In[ ]:


bert_outputs = bert_outputs[1]
classification_first = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(bert_outputs)
classification_model = tf.keras.Model([token_inputs, mask_inputs, segment_inputs], classification_first)
classification_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1.0e-5), loss=tf.keras.losses.BinaryCrossentropy(), metrics = [tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.FalsePositives(name='FalsePositive'), 'accuracy'])


# In[ ]:


classification_model.summary()


# In[ ]:


import tensorflow_addons as tfa
opt = tfa.optimizers.RectifiedAdam(lr=1.0e-5, weight_decay=0.0025)


# In[ ]:


def create_classification_bert():
  model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
  token_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_word_ids')
  mask_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_masks')
  segment_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_segment')
  bert_outputs = model([token_inputs, mask_inputs, segment_inputs])

  bert_outputs = bert_outputs[1]
  classification_first = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(bert_outputs)
  classification_model = tf.keras.Model([token_inputs, mask_inputs, segment_inputs], classification_first)

  classification_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1.0e-5), loss=tf.keras.losses.BinaryCrossentropy(), metrics = [tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.FalsePositives(name='FalsePositive'), 'accuracy'])
  return classification_model


# In[ ]:


classification_model = create_classification_bert()
  
hist = classification_model.fit(train_x, train_y, epochs=4, shuffle=True, batch_size=20, validation_data=(test_x, test_y))


# In[ ]:


print(hist.history['loss'])
print(hist.history['accuracy']) 
print(hist.history['precision'])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train accuracy')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val accuracy')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

####################################
fig, prec_ax = plt.subplots()

prec_ax.plot(hist.history['precision'], 'y', label='train precision')
prec_ax.plot(hist.history['val_precision'], 'r', label='val precision')

prec_ax.set_xlabel('epoch')
prec_ax.set_ylabel('precision')

prec_ax.legend(loc='upper left')

plt.show()


# In[ ]:


path = "/content/gdrive/My Drive/Colab Notebooks/News_detection"


# In[ ]:


os.getcwd()


# In[ ]:


classification_model.save_weights(path+"/News_detection_ver3(noprepro_multibert_title).h5")


# In[ ]:


def sentence_convert_data(data):
    global tokenizer
    tokens, masks, segments = [], [], []
    token = tokenizer.encode(data, max_length=SEQ_LEN, truncation=True, padding='max_length')
    
    num_zeros = token.count(0) 
    mask = [1]*(SEQ_LEN-num_zeros) + [0]*num_zeros 
    segment = [0]*SEQ_LEN

    tokens.append(token)
    segments.append(segment)
    masks.append(mask)

    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)
    return [tokens, masks, segments]

def information_evaluation_predict(sentence):
    data_x = sentence_convert_data(sentence)
    predict = classification_model.predict(data_x)
    predict_value = np.ravel(predict)
    predict_answer = np.round(predict_value,0).item()
    
    if predict_answer == 0:
      print("(정보가 있을 확률 : %.2f) 정보가 있는 문장입니다." % (1-predict_value))
    elif predict_answer == 1:
      print("(정보가 없을 확률 : %.2f) 정보가 없는 문장입니다." % predict_value)

