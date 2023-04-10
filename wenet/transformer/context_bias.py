#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: azhang
"""Encoder definition."""
from turtle import forward
from typing import Tuple

import torch
from typeguard import check_argument_types

from wenet.transformer.attention import MultiHeadedAttention
from wenet.transformer.attention import RelPositionMultiHeadedAttention
from wenet.transformer.convolution import ConvolutionModule
from wenet.transformer.embedding import PositionalEncoding
from wenet.transformer.embedding import RelPositionalEncoding
from wenet.transformer.embedding import NoPositionalEncoding
from wenet.transformer.transformer_encoder_layer import TransformerEncoderLayer, TransformerEncoder
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.transformer.subsampling import Conv2dSubsampling4
from wenet.transformer.subsampling import Conv2dSubsampling6
from wenet.transformer.subsampling import Conv2dSubsampling8
from wenet.transformer.subsampling import LinearNoSubsampling
from wenet.utils.common import get_activation
from wenet.utils.mask import make_pad_mask
from wenet.utils.mask import add_optional_chunk_mask
from wenet.mytransformer.transformer import Transformer
from wenet.mytransformer.attention import MultiHeadedAttentionPure


class BLSTM(torch.nn.Module):
    """
        Implementation of BLSTM Concatenation for sentiment classification task
    """

    def __init__(self, vocab_size, hidden_dim, num_layers, dropout=0.0):
        super(BLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = hidden_dim
        self.word_embedding = torch.nn.Embedding(
            self.vocab_size, self.embedding_dim)

        self.input_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim

        # sen encoder
        self.sen_rnn = torch.nn.LSTM(input_size=self.input_dim,
                                     hidden_size=self.hidden_dim,
                                     num_layers=num_layers,
                                     dropout=dropout,
                                     batch_first=True,
                                     bidirectional=True)


    def forward(self, sen_batch, sen_lengths):
        sen_batch = torch.clamp(sen_batch, 0)
        sen_batch = self.word_embedding(sen_batch)
        pack_seq = torch.nn.utils.rnn.pack_padded_sequence(
            sen_batch, sen_lengths.to('cpu').type(torch.int64), batch_first=True, enforce_sorted=False)
        _, last_state = self.sen_rnn(pack_seq)
        laste_h = last_state[0]
        laste_c = last_state[1]
        state = torch.cat([laste_h[-1, :, :], laste_h[0, :, :],
                          laste_c[-1, :, :], laste_c[0, :, :]], dim=-1)
        # state = torch.cat([laste_h[-1, :, :], laste_h[0, :, :]], dim=-1)
        # import pdb;pdb.set_trace()
        return state

class LSTM(torch.nn.Module):
    """
        Implementation of LSTM Concatenation for sentiment classification task
    """

    def __init__(self, vocab_size, hidden_dim, num_layers, dropout=0.0):
        super(LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = hidden_dim
        self.word_embedding = torch.nn.Embedding(
            self.vocab_size, self.embedding_dim)
        
        self.linear = torch.nn.Linear(hidden_dim * 2, hidden_dim * 4)

        self.input_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim

        # sen encoder
        self.sen_rnn = torch.nn.LSTM(input_size=self.input_dim,
                                     hidden_size=self.hidden_dim,
                                     num_layers=num_layers,
                                     dropout=dropout,
                                     batch_first=True,
                                     bidirectional=False)

    def forward(self, sen_batch, sen_lengths):
        sen_batch = torch.clamp(sen_batch, 0)
        sen_batch = self.word_embedding(sen_batch)
        pack_seq = torch.nn.utils.rnn.pack_padded_sequence(
            sen_batch, sen_lengths.to('cpu').type(torch.int64), batch_first=True, enforce_sorted=False)
        _, last_state = self.sen_rnn(pack_seq)
        
        laste_h = last_state[0]
        laste_c = last_state[1]
        state = torch.cat([laste_h[-1, :, :], laste_c[-1, :, :]], dim=-1)
        state = self.linear(state)
        # import pdb;pdb.set_trace()
        return state

class Transformer_extractor(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        output_size: int,
        layernorm_shape: int,
        attention_head: int,
        linear_units: int,
        num_layers: int,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = hidden_dim
        self.layernorm_shape = layernorm_shape
        self.dropout_rate = dropout_rate
        self.output_size = output_size
        self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.input_layer = LinearNoSubsampling(
            idim=self.embedding_dim,
            odim=self.output_size,
            dropout_rate=self.dropout_rate,
            pos_enc_class=PositionalEncoding(self.embedding_dim, self.dropout_rate)
        )
        self.encoders = torch.nn.ModuleList([TransformerEncoderLayer(
                                                self.output_size,
                                                MultiHeadedAttention(attention_head, self.output_size, attention_dropout_rate),
                                                PositionwiseFeedForward(self.output_size, linear_units, dropout_rate),
                                                self.dropout_rate)
                                            for _ in range(num_layers)
                                            ])
        self.after_norm = torch.nn.LayerNorm(self.layernorm_shape)
        self.linear = torch.nn.Linear(self.output_size, self.output_size * 4)

    def forward(self, X, length):
        X = torch.clamp(X, 0)
        sos = torch.ones(length.size(0), 1).to(X.device)
        X = torch.cat([sos, X], dim=1)
        length = length + torch.ones(length.size(0)).to(X.device)
        X = self.embed(X.long())
        masks = ~make_pad_mask(length, X.size(1)).unsqueeze(1).to(X.device)
        X, pos_emb, masks = self.input_layer(X, masks)
        for layer in self.encoders:
            X, masks, _, _  = layer(X, masks, pos_emb)
        X = self.after_norm(X)
        X = self.linear(X)
        return X[:, 0, :]


class ContextBias(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        vocab_size: int,
        embedding_size: int,
        num_layers: int = 2,
        attention_heads: int = 4,
        linear_units: int = 512,
        num_block: int = 4,
        dropout_rate: float = 0.0,
        bias_encoder_type: str = "linear",
        bias_encoder: bool = True,
        context_extractor: str = "BLSTM"
    ):
        assert check_argument_types()
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.attention_heads = attention_heads
        self.linear_units = linear_units
        self.num_blocks = num_block
        self.dropout_rate = dropout_rate
        self.encoder_type = bias_encoder_type
        self.if_bias_encoder = bias_encoder
        self.context_extractor = context_extractor
        if self.context_extractor == 'BLSTM':
            self.context_extractor = BLSTM(
                self.vocab_size, self.embedding_size, self.num_layers)
        elif self.context_extractor == 'LSTM':
            self.context_extractor = LSTM(
                self.vocab_size, self.embedding_size, self.num_layers
            )
        elif self.context_extractor == 'transformer':
            self.context_extractor = Transformer_extractor(
                vocab_size=self.vocab_size,
                hidden_dim=self.embedding_size,
                output_size=self.embedding_size,
                layernorm_shape=self.embedding_size,
                attention_head=8,
                linear_units=self.embedding_size * 4,
                num_layers=3
                )
        if self.encoder_type == 'transformer':
            self.context_encoder = TransformerEncoder(
                input_size=self.embedding_size * 4,
                output_size=self.embedding_size,
                attention_heads=self.attention_heads,
                linear_units=self.linear_units,
                num_blocks=self.num_blocks,
                dropout_rate=self.dropout_rate,
                positional_dropout_rate=0.0,
                attention_dropout_rate=0.0,
                input_layer="linear",
                pos_enc_layer_type="no_pos",
                normalize_before=True,
                concat_after=False,
                static_chunk_size=0,
                )
        elif self.encoder_type == 'linear':
            self.context_encoder = torch.nn.Sequential(torch.nn.Linear(self.embedding_size * 4,self.embedding_size),torch.nn.LayerNorm(self.embedding_size))
        elif self.encoder_type == 'mytransformer':
            self.context_encoder = Transformer(
                args=None,
                input_dim=self.embedding_size * 4,
                output_dim=self.embedding_size,
                attention_dim=self.embedding_size,
                attention_heads=self.attention_heads,
                linear_units=self.linear_units,
                num_blocks=self.num_blocks,
                dropout_rate=0.0,
                positional_dropout_rate=0.0,
                attention_dropout_rate=0.0,
                input_layer="linear",
                pos_enc_class="abs-enc",
                normalize_before=True,
                concat_after=False,
                positionwise_layer_type="linear",
                positionwise_conv_kernel_size=1,
                chunk_size=-1,
                left_chunks=-1,
            )
        if self.encoder_type == 'mytransformer':
            self.encoder_bias = MultiHeadedAttentionPure(
                n_head=self.attention_heads,
                n_feat=self.embedding_size,
                dropout_rate=0.0,
                chunk_size=-1,
                left_chunks=-1,
                pos_enc_class=None,
            )
        else:
            self.encoder_bias = MultiHeadedAttention(
                n_head=self.attention_heads,
                n_feat=self.embedding_size,
                dropout_rate=0.0
            )
        # self.encoder_norm = torch.nn.LayerNorm(self.embedding_size)
        # self.encoder_bias_norm = torch.nn.LayerNorm(self.embedding_size)
        # self.encoder_ffn = torch.nn.Linear(self.embedding_size * 2, self.output_size)
    

    
    def forward(self, context_list, context_lengths, h_enc):
        bias_vector = self.context_extractor(context_list,context_lengths)
        bias_lengths = torch.tensor([bias_vector.shape[0]],dtype=torch.int32,device=bias_vector.device)
        # if self.encoder_type == 'transformer':
        #     bias_hidden, bias_mask = self.context_encoder(
        #         bias_vector.unsqueeze(0),bias_lengths
        #     )
        # elif self.encoder_type == 'linear':
        bias_hidden = self.context_encoder(bias_vector.unsqueeze(0))
        # elif self.encoder_type == 'mytransformer':
        #     bias_hidden, ilens, bias_mask = self.context_encoder(
        #         bias_vector.unsqueeze(0),bias_lengths
        #     )
        bias_hidden = bias_hidden.expand(h_enc.shape[0],-1,-1)
        
        # if self.encoder_type == 'mytransformer':
        #     h_enc_bias = self.encoder_bias(h_enc, bias_hidden, bias_hidden)
        # else:
        h_enc_bias,_ = self.encoder_bias(h_enc, bias_hidden, bias_hidden)
        encoder_norm_out = self.encoder_norm(h_enc)
        encoder_bias_norm_out = self.encoder_bias_norm(h_enc_bias)
        h_enc_concat = torch.cat(
            [encoder_norm_out, encoder_bias_norm_out],dim=-1
        )

        return h_enc + self.encoder_ffn(h_enc_concat)
    
    def forword_common(self, context_list, context_lengths, h_enc):
        bias_vector = self.context_extractor(context_list,context_lengths)
        bias_lengths = torch.tensor([bias_vector.shape[0]],dtype=torch.int32,device=bias_vector.device)
        # if self.encoder_type == 'transformer':
        #     bias_hidden, bias_mask = self.context_encoder(
        #         bias_vector.unsqueeze(0),bias_lengths
        #     )
        # elif self.encoder_type == 'linear':
        bias_hidden = self.context_encoder(bias_vector.unsqueeze(0))
        # elif self.encoder_type == 'mytransformer':
        #     bias_hidden, ilens, bias_mask = self.context_encoder(
        #         bias_vector.unsqueeze(0),bias_lengths
        #     )
        bias_hidden = bias_hidden.expand(h_enc.shape[0],-1,-1)
        return bias_hidden
    def forward_encoder(self, bias_hidden, h_enc):
        # if self.encoder_type == 'mytransformer':
        #     h_enc_bias = self.encoder_bias(h_enc, bias_hidden, bias_hidden)
        # else:
        h_enc_bias,_ = self.encoder_bias(h_enc, bias_hidden, bias_hidden)
        encoder_norm_out = self.encoder_norm(h_enc)
        encoder_bias_norm_out = self.encoder_bias_norm(h_enc_bias)
        h_enc_concat = torch.cat(
            [encoder_norm_out, encoder_bias_norm_out],dim=-1
        )
        
        return h_enc + self.encoder_ffn(h_enc_concat)
    
    def forward_bias_hidden(self, context_list, context_lengths):
        bias_vector = self.context_extractor(context_list, context_lengths)
        bias_lengths = torch.tensor([bias_vector.shape[0]],dtype=torch.int32,device=bias_vector.device)
        # if self.encoder_type == 'transformer':
        #     bias_hidden, bias_mask = self.context_encoder(
        #         bias_vector.unsqueeze(0),bias_lengths
        #     )
        # elif self.encoder_type == 'linear':
        bias_hidden = self.context_encoder(bias_vector.unsqueeze(0))
        # elif self.encoder_type == 'mytransformer':
        #     bias_hidden, ilens, bias_mask = self.context_encoder(
        #         bias_vector.unsqueeze(0),bias_lengths
        #     )
        return bias_hidden
        
    def forword_context_linear(self, context_list, context_lengths):
        bias_vector = self.context_extractor(context_list,context_lengths)
        bias_hidden = self.context_encoder(bias_vector.unsqueeze(0))
        return bias_hidden

    def forward_encoder_layer(self, bias_hidden, h_enc):
        bias_hidden = bias_hidden.expand(h_enc.shape[0],-1,-1)
        # if self.encoder_type == 'mytransformer':
        #     h_enc_bias = self.encoder_bias(h_enc, bias_hidden, bias_hidden)
        # else:
        h_enc_bias, _ = self.encoder_bias(h_enc, bias_hidden, bias_hidden)
        return h_enc_bias


    def forword_context_encoder(self, bias_hidden, encoder_out):
        encoder_out_context, _ = self.encoder_bias(encoder_out, bias_hidden, bias_hidden)
        encoder_norm_out = self.encoder_norm(encoder_out)
        encoder_bias_norm_out = self.encoder_bias_norm(encoder_out_context)
        encoder_out_cat = torch.cat([encoder_norm_out, encoder_bias_norm_out], dim=-1)
        return encoder_out + self.encoder_ffn(encoder_out_cat)