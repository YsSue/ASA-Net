import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

checkpoint_path='train_log/pre_icme_5/model-10800'#your ckpt path
#reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
reader = tf.train.NewCheckpointReader(checkpoint_path)

var_to_shape_map=reader.get_variable_to_shape_map()

params={}
for key in var_to_shape_map:
    str_name = key
    '''
    if not str_name.startswith('EMA'):
        str_name=str_name.replace('/Momentum','')
        str_name=str_name.replace('/AccumGrad','')
        str_name=str_name.replace('learning_rate','l')
        str_name=str_name.replace('global_step','l')
        str_name=str_name+':0'
        print('tensor_name:' , str_name)
    '''
    str_name='att/'+str_name
    params[str_name]=reader.get_tensor(key)
    print('tensor_name:' , str_name)
# save npy

np.save('pretrain_model_for_att.npy',params)
print('save npy over...')