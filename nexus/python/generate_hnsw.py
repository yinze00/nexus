# generate hnsw_graph
# author junwei.wang@zju.edu.cn
from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops # type: ignore

import faiss
from argparse import ArgumentParser

import sys

class HNSWGraphGenerator(object):
    def __init__(self) -> None:
        self.parser = ArgumentParser()
    
    def addOptions(self):
        self.parser.add_argument('--hnsw_path', type=str, default='nexus/data/hnsw_model/data/hnsw_1000000.dat')
        self.parser.add_argument('--output_path', type=str, default='nexus/data/hnsw_model')
    
    def parse(self):
        self.addOptions()
        options = self.parser.parse_args()
        self.options = options
        
        self.hnsw_path = self.options.hnsw_path
        self.output_path = self.options.output_path
        
        return True
    

    def load_index(self):
        self.index = faiss.downcast_index(faiss.read_index(self.hnsw_path))
        self.entry_point = self.index.hnsw.entry_point
        self.max_level = self.index.hnsw.max_level
        self.d = self.index.d
        print(self.max_level, self.d)
    
    
    def generate(self):
        
        g = tf.Graph()
        with g.as_default():
            hnsw_module = tf.load_op_library('/home/yinze/dev/zenith/nexus/nexus/cc/nexus_ops_defs.so')

            # graph feeds 
            user_emb    = tf.compat.v1.placeholder(tf.float32, name='user_emb')
            # entry_point = tf.compat.v1.placeholder(tf.int32, name='entry_point')
            hints       = tf.compat.v1.placeholder(tf.int32, name='hints')
            
            entry_point = hnsw_module.request_init_op(index_name="hnsw_demo")
            
            neis = hnsw_module.gather_neighbors_op(entry_point, level=self.max_level, index_name="hnsw_demo")
                        
            for level in range(self.max_level , -1 , -1):
                level = level - 1
                embs = hnsw_module.gather_embeddings_op(neis, index_name="hnsw_demo", dim=self.d)
                sims = hnsw_module.gemv_op(embs, user_emb)
                
                entry_point_of_next, _ = hnsw_module.indirect_sort_and_topk_op(neis, sims, topk=1000)
                
                if level:
                    neis = hnsw_module.gather_neighbors_op(entry_point_of_next, level=level, index_name="hnsw_demo")
                
            labels, scores = hnsw_module.result_construct_op(entry_point_of_next, _, index_name="hnsw_demo")
                
            with tf.control_dependencies([labels, scores]):
                done = tf.no_op(name="done")
            
            
            
        return g

if __name__ == '__main__':
    
    print('tensorflow version : ', tf.__version__)
    gen = HNSWGraphGenerator()
    if not gen.parse():
        sys.exit(-1)

    gen.load_index()
        
    g =     gen.generate()
    g_def = g.as_graph_def()

    with open(gen.output_path + "/hnsw_graph.pbtxt", 'w') as f:
        f.write(str(g_def))
