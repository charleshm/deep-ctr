#!usr/bin/python
# -*- coding: gbk -*-

import tensorflow as tf

tf.set_random_seed(2018)

class DeepFM:
    def __init__(self, feature_size, factor_size, field_size,
                 deep_layers=[400, 400], 
                 deep_layers_activation=tf.nn.relu):
        self.feature_size = feature_size
        self.factor_size  = factor_size
        self.field_size   = field_size
        self.deep_layers  = deep_layers
        self.deep_layers_activation = deep_layers_activation
    
    def first_order_part(self, sparse_id, sparse_value):
        with tf.variable_scope("first-order"):
            W    = tf.get_variable("weight",(self.feature_size, 1), \
                    initializer=tf.random_normal_initializer(0.0, 0.01))
            y_first_order = tf.nn.embedding_lookup(W, sparse_id) # None * F * 1
            y_first_order = tf.reduce_sum(tf.multiply(y_first_order, sparse_value), 1)  # None * 1

            return y_first_order
    
    def second_order_part(self, sparse_id, sparse_value):
        with tf.variable_scope("second-order"):
            V = tf.get_variable("weight",(self.feature_size, self.factor_size), \
                    initializer=tf.random_normal_initializer(0.0, 0.01))
            self.embeddings1 = tf.nn.embedding_lookup(V, sparse_id)
            self.embeddings1 = tf.multiply(self.embeddings1, sparse_value) # None * F * K

            # 平方和 
            sum_squared_part = tf.square(tf.reduce_sum(self.embeddings1, 1)) # None * K
            # 和平方
            squared_sum_part = tf.reduce_sum(tf.square(self.embeddings1), 1) # None * K


            y_second_order   = 0.5 * tf.subtract(sum_squared_part, squared_sum_part)
            return y_second_order
    
    def mvm_part(self, sparse_id, sparse_value):
        with tf.variable_scope("mvm"):
            W    = tf.get_variable("core_embedding",(self.feature_size, self.factor_size), \
                    initializer=tf.random_normal_initializer(0.5, 0.01))
            bias = tf.get_variable("padding_bias", (self.field_size, self.factor_size), \
                    initializer=tf.random_normal_initializer(0.5, 0.01))

            self.embeddings2 = tf.nn.embedding_lookup(W, sparse_id) 
            self.embeddings2 = tf.multiply(self.embeddings2, sparse_value)   # None * F * K

            all_order  = tf.add(self.embeddings2, bias)
            mvm_func   = all_order[:,0,:]    # None * 1 * K
            for i in range(1, self.field_size):
                mvm_func = tf.multiply(mvm_func, all_order[:,i,:])
            
            mvm_func   =  tf.reshape(mvm_func, shape=[-1, self.factor_size]) # None * K

            return mvm_func

    def deep_part(self):
        with tf.variable_scope("deep-part"):
            y_deep = tf.concat([self.embeddings1, self.embeddings2], axis=1)
            y_deep = tf.reshape(y_deep, shape=[-1, \
                            2 * self.field_size * self.factor_size]) # None * (F*K)
            for i in range(0, len(self.deep_layers)):
                y_deep = tf.contrib.layers.fully_connected(y_deep, self.deep_layers[i], \
                            activation_fn=self.deep_layers_activation, scope = 'fc%d' % i)
            
            return y_deep

        
    def forward(self, sparse_id, sparse_value):
        sparse_value   = tf.expand_dims(sparse_value, -1)

        y_first_order  = self.first_order_part(sparse_id, sparse_value)
        y_second_order = self.second_order_part(sparse_id, sparse_value)
        mvm            = self.mvm_part(sparse_id, sparse_value)
        y_deep         = self.deep_part()

        with tf.variable_scope("deep-mvm"):
            deep_out    = tf.concat([y_first_order, y_second_order, mvm, y_deep], axis=1)
            deep_out    = tf.contrib.layers.fully_connected(deep_out, 1, \
                activation_fn=tf.nn.sigmoid, scope = 'deepmvm_out')

            return tf.reduce_sum(deep_out, axis=1)
            
