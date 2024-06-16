
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )


# g++ -std=c++14  -D_GLIBCXX_USE_CXX11_ABI=0 -DEIGEN_MAX_ALIGN_BYTES=64 -shared -march=native kernel/*.cc ops/*.cc -o zero_out.so -fPIC -O2 -I/usr/local/include/tensorflow -l:libtensorflow_framework.so
g++ -std=c++14  -D_GLIBCXX_USE_CXX11_ABI=0 -DEIGEN_MAX_ALIGN_BYTES=64 -DPROTOBUF_VERSION=3008000 -DPROTOBUF_MIN_PROTOC_VERSION=3008000 -shared -march=native kernel/*.cc ops/*.cc -o zero_out.so -fPIC -O2 -I/home/yinze/libtf1.15/include -L/home/yinze/libtf1.15/lib/ -l:libtensorflow_framework.so


