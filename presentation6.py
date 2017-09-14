import tensorflow as tf
from tensorflow.python.client import device_lib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Creates a graph.



#a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#c = tf.matmul(a, b)
## Creates a session with log_device_placement set to True.
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
## Runs the op.
#print(sess.run(c))



#config = tf.ConfigProto(device_count={"CPU": 2},
#                        inter_op_parallelism_threads=2,
#                        intra_op_parallelism_threads=1)
#sess = tf.Session(config=config)

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_availabile_cpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']

print("CPUS " , get_availabile_cpus())
print("GPUS " , get_available_gpus())



