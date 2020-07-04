import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def scaled_dot_product_attention(query, key, value, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """
  #print(f"q: {q.shape}, k:{k.shape}, v:{v.shape}")
  matmul_qk = tf.matmul(query, key, transpose_b=True)

  # scale matmul_qk
  depth = tf.cast(tf.shape(key)[-1], tf.float32)
  logits = matmul_qk / tf.math.sqrt(depth)

  # add the mask to zero out padding tokens
  if mask is not None:
    logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k)
  attention_weights = tf.nn.softmax(logits, axis=-1)

  output = tf.matmul(attention_weights, value)

  return output


class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, name="multi_head_attention"):
    super(MultiHeadAttention, self).__init__(name=name)
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.query_dense = tf.keras.layers.Dense(units=d_model)
    self.key_dense = tf.keras.layers.Dense(units=d_model)
    self.value_dense = tf.keras.layers.Dense(units=d_model)

    self.dense = tf.keras.layers.Dense(units=d_model)

  def split_heads(self, inputs, batch_size):
    inputs = tf.reshape(
        inputs, shape=(batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(inputs, perm=[0, 2, 1, 3])

  def call(self, query, key, value, mask):
    batch_size = tf.shape(query)[0]

    # linear layers
    query = self.query_dense(query)
    key = self.key_dense(key)
    value = self.value_dense(value)

    # split heads
    query = self.split_heads(query, batch_size)
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)

    # scaled dot-product attention
    print(f"Q:{query.shape}, K:{key.shape}, V:{value.shape}")
    scaled_attention = scaled_dot_product_attention(query, key, value, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    # concatenation of heads
    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))

    # final linear layer
    outputs = self.dense(concat_attention)

    return outputs

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.self_attn = MultiHeadAttention(d_model = d_model, num_heads=num_heads)
    self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    self.ffn1 = tf.keras.layers.Dense(dff, activation='relu')
    self.dropout = tf.keras.layers.Dropout(rate)
    self.ffn2 = tf.keras.layers.Dense(d_model)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)
    
    
  def call(self, tgt, enc_output, training=True,
           look_ahead_mask=None, padding_mask=None):
      
    tgt = tf.transpose(tgt, [1, 0, 2])
    enc_output = tf.transpose(enc_output, [1, 0, 2])
      
    #Changed First layernorm then  masked attn
    tgt = self.layernorm1(tgt)
    print(f"TGT:{tgt.shape}")
    tgt2 = self.self_attn(tgt, tgt, tgt, mask=look_ahead_mask)
    #print(tgt2.shape)
    tgt = tgt + self.dropout1(tgt2)
    
    #print(enc_output.shape)
    #LayerNorm then decoder attn
    tgt = self.layernorm2(tgt)
    print(f"TGT:{tgt.shape}, ENC: {enc_output.shape}")
    tgt2 = self.mha(tgt, enc_output, enc_output, mask=None)
    print(f"target: {tgt.shape}, enc_op:{enc_output.shape}")
    tgt = tgt + self.dropout2(tgt2)
    
    #LayerNorm then FFN
    tgt = self.layernorm3(tgt)
    tgt2 = self.ffn2(self.dropout(self.ffn1(tgt),training))
    tgt = tgt + self.dropout3(tgt2)
    
    tgt = tf.transpose(tgt, [1, 0, 2])

    return tgt



class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
    
  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):

    attention_weights = {}

    for i in range(self.num_layers):
      x = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)
    
    # x.shape == (batch_size, target_seq_len, d_model)
    return x


