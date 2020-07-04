import tensorflow as tf
import tensorflow.keras.layers as L

class WordAndPositionalEmbedding(tf.keras.layers.Layer):
    r"""
    A :class: Learned word embeddings and position
    embeddings for input tokens. Each token is mapped to a fixed dimensional
    word embedding; and corresponding positional embedding based on its index.
    These are summed together followed by layer normalization and an optional
    dropout.
    Parameters
    ----------
    vocab_size: int
        Size of token vocabulary.
    hidden_size: int
        Size of token embedding vectors.
    max_caption_length: int, optional (default = 30)
        Maximum length of input captions; this is used to create a fixed
        positional embedding lookup table.
    dropout: float, optional (default = 0.1)
        Dropout probability for final dropout applied after layer normalization.
    padding_idx: int, optional (default = 0)
        Token index of ``[PAD]`` token, word embedding for these tokens will
        be a vector of zeroes (and not trainable).
    """

    def __init__(self, 
                 vocab_size: int, 
                 hidden_size:int, 
                 max_caption_length: int = 30, 
                 rate:float = 0.0,
                 padding_idx: int=0):
        super(WordAndPositionalEmbedding, self).__init__()

        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

        self.words = L.Embedding(vocab_size, hidden_size, mask_zero=True)

        self.positions = L.Embedding(max_caption_length, hidden_size)
        self.layer_norm = L.LayerNormalization(
            epsilon=1e-8,
        )

        self.dropout = L.Dropout(rate = rate)


    def call(self, tokens):
        
        r"""
        Get combined word and positional embeddings for input tokens.
        Parameters
        ----------
        tokens: torch.Tensor
            A tensor of shape ``(batch_size, max_caption_length)`` containing
            a batch of caption tokens, with values in ``[0, vocab_size)``.
        Returns
        -------
        torch.Tensor
            A tensor of shape ``(batch_size, max_caption_length, hidden_size)``
            containing corresponding token embeddings.
        """
        print(tokens.shape)
        position_indices = self._create_position_indices(tokens)
        print(position_indices.shape)

        word_embeddings = self.words(tokens)
        positional_embeddings = self.positions(position_indices)


        embeddings = self.layer_norm(word_embeddings + positional_embeddings)
        embeddings = self.dropout(embeddings)


        token_mask = tf.expand_dims(tokens != self.padding_idx, -1)


        embeddings = embeddings * tf.cast(token_mask, dtype=embeddings.dtype)
        return embeddings

    def _create_position_indices(self, tokens):

        # Create position indices of the same size as token indices.
        batch_size, max_caption_length = tokens.shape
        positions = tf.range(
            max_caption_length, dtype=tokens.dtype
        )
        # shape: (batch_size, max_caption_length)
        positions = tf.broadcast_to(tf.expand_dims(positions, 0), [batch_size, max_caption_length])        
        return positions