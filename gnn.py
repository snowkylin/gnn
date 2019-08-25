import tensorflow as tf


class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, activation=None, **kwargs):
        self.output_dim = output_dim
        self.activation = activation
        super().__init__(**kwargs)

    def build(self, input_shape):   # [batch_size, num_vertices, num_vertices], [batch_size, num_vertices, num_features]
        A_shape, H_shape = input_shape
        self.num_vertices = A_shape[1].value
        self.W = self.add_weight(   # [num_features, output_dim]
            name='W',
            shape=[H_shape[2].value, self.output_dim]
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """
        :param inputs:  A for adjacent matrix [batch_size, num_vertices, num_vertices] (should be normalized in advance)
                        H for features [batch_size, num_vertices, num_features]
        """
        A, H = inputs[0], inputs[1]
        # A * H * W [batch_size, num_vertices, num_vertices] * [batch_size, num_vertices, num_features] * [num_features, output_dim]
        # see https://www.tensorflow.org/api_docs/python/tf/tensordot and https://www.machenxiao.com/blog/tensordot
        # for tf.tensordot()
        H_next = tf.tensordot(tf.matmul(A, H), self.W, axes=[2, 0])
        if self.activation is not None:
            H_next = self.activation(H_next)
        return H_next


class GATLayer(tf.keras.layers.Layer):
    # reference: https://github.com/danielegrattarola/keras-gat/blob/master/keras_gat/graph_attention_layer.py
    def __init__(self, output_dim, activation=None, **kwargs):
        self.output_dim = output_dim
        self.activation = activation
        super().__init__(**kwargs)

    def build(self, input_shape):
        A_shape, H_shape = input_shape
        self.W = self.add_weight(  # [output_dim, num_features]
            name='W',
            shape=[H_shape[2].value, self.output_dim]
        )
        # a = [a_1, a_2]
        self.a_1 = self.add_weight(name='a_1', shape=[self.output_dim, 1])
        self.a_2 = self.add_weight(name='a_2', shape=[self.output_dim, 1])

    def call(self, inputs, **kwargs):
        A, H = inputs[0], inputs[1]
        # [batch_size, num_vertices, num_features] * [num_features, output_dim]
        H_ = tf.tensordot(H, self.W, axes=[2, 0])   # [batch_size, num_vertices, output_dim]
        e = tf.nn.leaky_relu(
            tf.tensordot(H_, self.a_1, axes=[2, 0]) + tf.transpose(tf.tensordot(H_, self.a_2, axes=[2, 0]), perm=[0, 2, 1]),
            alpha=0.2
        )  # [batch_size, num_vertices, num_vertices]
        A = tf.cast(tf.math.greater(A, 0.0), dtype=tf.float32)
        alpha = tf.nn.softmax(e * A)
        H_next = tf.matmul(alpha, H_)
        if self.activation is not None:
            return self.activation(H_next)
        else:
            return H_next


class MultiHeadGATLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, num_heads, activation, aggregation, **kwargs):
        """
        :param aggregation: 'concat' or 'average'
        """
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.activation = activation
        self.aggregation = aggregation
        self.layers = [GATLayer(output_dim, activation=None) for _ in range(num_heads)]
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        A, H = inputs[0], inputs[1]
        H_next_list = [self.layers[i](A, H) for i in self.num_heads]
        if self.aggregation == 'concat':
            return self.activation(tf.concat(H_next_list, axis=-1))
        else:
            return self.activation(tf.reduce_mean(tf.stack(H_next_list, axis=-1), axis=-1))