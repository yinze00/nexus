# generate hnsw_graph
# author junwei.wang@zju.edu.cn
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops # type: ignore

import faiss
from argparse import ArgumentParser

import sys

class HNSWGraphGenerator(object):
    def __init__(self) -> None:
        self.parser = ArgumentParser()
    
    def addOptions(self):
        self.parser.add_argument('--hnsw_path', type=str, default='models/hnsw_model/data/hnsw_1000000.dat')
        self.parser.add_argument('--output_path', type=str, default='models/hnsw_model/')
    
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
    
    
    # tf model without hints 
    def generate(self):
        
        g = tf.Graph()
        with g.as_default():
            hnsw_module = tf.load_op_library('/home/yinze/dev/ha3/lib/libha3_suez_turing_agg_opdef.so')

            # graph feeds 
            user_emb    = tf.placeholder(tf.float32, name='user_emb')
            entry_point = tf.placeholder(tf.int32, name='entry_point')
            hints       = tf.placeholder(tf.int64, name='hints')
            
            neis = tf.gather_neighbors_op(entry_point)
                        
            for level in range(self.max_level, 0, -1):
                embs = tf.gather_embeddings_op(neis)
                sims = tf.matmul_op(embs, user_emb)
                
                
                # neis = tf.gather_neighbors_op(hints)
            
            
        return g

if __name__ == '__main__':
    gen = HNSWGraphGenerator()
    if not gen.parse():
        sys.exit(-1)
        
    g =     gen.generate()
    g_def = g.as_graph_def()

    with open(gen.output_path + "/hnsw_graph.pbtxt", 'w') as f:
        f.write(str(g_def))
