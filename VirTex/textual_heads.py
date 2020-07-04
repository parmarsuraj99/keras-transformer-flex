import tensorflow as tf
import tensorflow.keras.layers as L

from transformer import *
from embeddings import *

class TextualHead(tf.keras.layers.Layer):
    r"""
    Base class for all textual heads. All child classes can simply inherit
    from :class:`~torch.nn.Module`, however this is kept here for uniform
    type annotations.
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
    ):
        super(TextualHead, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    @property
    def textual_feature_size(self):
        
        r"""
        Size of the last dimension of output from forward pass; typically same
        as :attr:`hidden_size` for most modules. This property is used to add
        more modules on top of this.
        """
        return self.hidden_size


class LinearTextualHead(TextualHead):
    r"""
    Textual head containing a single linear layer projecting from textual
    feature size to output vocabulary size.
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
    ):
        super(LinearTextualHead, self).__init__(vocab_size, hidden_Size)
        self.output = L.Dense(vocab_size)

    def call(self,
             caption_tokens,
             caption_lengths,
             visual_features):
        
        output_logits = self.output(visual_features)
        return output_logits

class TransformerTextualHead(TextualHead):
    def __init__(
        self,
        vocab_size: int,
        hidden_size:int,
        num_layers: int,
        attention_heads: int,
        feedforward_size: int,
        dropout: float = 0.1,
        norm_type:str="pre",
        padding_idx: int=0,
        max_caption_length: int = 30
    ):
        super().__init__(vocab_size, hidden_size)
        self.num_layers = num_layers
        self.attention_heads =attention_heads
        self.feedforward_size = feedforward_size
        self.dropout = dropout
        self.padding_idx = padding_idx

        self.embedding = WordAndPositionalEmbedding(
            self.vocab_size,
            self.textual_feature_size,
            max_caption_length = max_caption_length,
            rate = dropout,
        )
        LayerClass = (
            DecoderLayer
        )
        _layer = DecoderLayer(
            self.textual_feature_size,
            self.attention_heads,
            dff = self.feedforward_size,
            rate = dropout
        )
        
        self.encoder = Decoder(self.num_layers, self.textual_feature_size,
            self.attention_heads,
            dff = self.feedforward_size,
            rate = dropout)
        """
        self.encoder = DecoderLayer(
            self.textual_feature_size,
            self.attention_heads,
            dff = self.feedforward_size,
            rate = dropout)
        """

        self.outputL = L.Dense(vocab_size)
        #self.output.weight = self.embedding.words.weight

    def call(self,
             caption_tokens,
             caption_lengths,
             visual_features,
             training: bool
             ):
        batch_size, max_caption_length = caption_tokens.shape
        print(max_caption_length)

        ones = tf.ones_like(caption_tokens)
        caption_mask = tf.expand_dims(caption_lengths, 1) < tf.cumsum(ones, 1)

        caption_embeddings = self.embedding(caption_tokens)

        unidirectional_mask = self._generate_future_mask(max_caption_length)

        print("cap_vis_mask:", caption_embeddings.shape, visual_features.shape, unidirectional_mask.shape)

        caption_embeddings = tf.transpose(caption_embeddings, [1, 0, 2])
        visual_features = tf.transpose(visual_features, [1, 0, 2])
        print(caption_embeddings.shape, visual_features.shape)

        textual_features = self.encoder(caption_embeddings,
                                        visual_features,
                                        look_ahead_mask=unidirectional_mask, 
                                        padding_mask=caption_mask,
                                        training=training)
        textual_features = tf.transpose(textual_features, [1, 0, 2])
        op = self.outputL(textual_features)

        return op

    def  _generate_future_mask(self, size:int):

        mask = tf.linalg.band_part(
            tf.ones([size, size]), 0, -1
        )
        return mask
