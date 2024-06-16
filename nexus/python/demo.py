import tensorflow as tf
import tensorboard as tb

from tensorflow.python.ops import control_flow_ops # type: ignore


print(tf.__version__)

g = tf.Graph()

with g.as_default():
    # 创建一个可训练的变量
    my_variable = tf.Variable(0.0, name="my_variable")

    # tf.Variable()

    # 在某个操作中使用这个变量
    update_op = my_variable.assign_add(1.0)

    # 初始化变量
    init_op = tf.compat.v1.global_variables_initializer()
    
gdef = g.as_graph_def()
outputName = 'demo'

with open("./%s.pbtxt" % outputName, 'w') as f:
    f.write(str(gdef))
    
# writer = tf.summary.FileWriter("./%s.tensorboard" % outputName, gdef)