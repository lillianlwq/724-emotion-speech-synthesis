from keras.layers import MultiHeadAttention

from Blocks import *


class Transformer(nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: InputEmbedding,
                 trg_embed: InputEmbedding,
                 src_pos: PositionalEncoding,
                 trg_pos: PositionalEncoding,
                 projection: ProjectionLayer,
                 ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.src_pos = src_pos
        self.trg_pos = trg_pos
        self.projection = projection

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_outputs, src_mask, trg, trg_mask):
        trg = self.trg_embed(trg)
        trg = self.trg_pos(trg)
        return self.decoder(trg, encoder_outputs, src_mask, trg_mask)

    def project(self, x):
        return self.projection(x)

    def build_transformer(self,
                          src_vocab_size: int,
                          trg_vocab_size: int,
                          src_seq_len: int,
                          trg_seq_len: int,
                          d_model: int = 512,
                          n: int = 6,
                          heads: int = 8,
                          dropout: float = 0.1,
                          d_ff: int = 2048,
                          ):
        # create embedding layers
        src_embed = InputEmbedding(d_model, src_vocab_size)
        trg_embed = InputEmbedding(d_model, trg_vocab_size)

        # create the positional encoding layers
        src_pos = PositionalEncoding(d_model, src_seq_len, dropout=dropout)
        trg_pos = PositionalEncoding(d_model, trg_seq_len, dropout=dropout)

        # Create the encoder blocks
        encoder_blocks = []
        for _ in range(n):
            encoder_self_attention_block = MultiHeadAttention(d_model, heads, dropout=dropout)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
            encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
            encoder_blocks.append(encoder_block)

        # create decoder blocks
        decoder_blocks = []
        for _ in range(n):
            decoder_self_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout=dropout)
            decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout=dropout)
            decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout=dropout)
            decoder_block = DecoderBlock(decoder_self_attention_block,
                                         decoder_cross_attention_block,
                                         decoder_feed_forward_block,
                                         dropout=dropout)
            decoder_blocks.append(decoder_block)

        # create the encoder and decoder
        encoder = Encoder(nn.ModuleList(encoder_blocks))
        decoder = Decoder(nn.ModuleList(decoder_blocks))

        #create the projection layer
        projection = ProjectionLayer(d_model, trg_vocab_size)

        transformer = Transformer(encoder, decoder, src_embed, trg_embed, src_pos, trg_pos, projection)

        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return transformer
                

