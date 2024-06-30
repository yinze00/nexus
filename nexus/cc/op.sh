TF_CFLAGS=( $(python3.7 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3.7 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

op_name=nexus_ops_defs

g++ -std=c++14 -shared ops/*.cc -o ${op_name}.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2

# python3.7 -c 'import tensorflow as tf; tf.load_op_library("./nexus_ops_def.so")'