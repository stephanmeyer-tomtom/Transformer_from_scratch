from torch import Tensor
import torch.nn.functional as F
import torch
from torch import nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int, add_mask: bool):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k, add_mask) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        layer_outputs = []
        for h in self.heads:
            layer_outputs.append(h(query, key, value))
        concat_out = torch.cat(layer_outputs, dim=1)
        lin_output = self.linear(concat_out)
        return lin_output


class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int, add_mask: bool):
        super().__init__()
        self.fc_layer_q = nn.Linear(dim_in, dim_q)
        self.fc_layer_k = nn.Linear(dim_in, dim_k)
        self.fc_layer_v = nn.Linear(dim_in, dim_k)
        self.add_mask = add_mask

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        query_val = self.fc_layer_q(query)
        key_val = self.fc_layer_k(key)
        value_val = self.fc_layer_v(value)
        return scaled_dot_product_attention(query_val, key_val, value_val, self.add_mask)


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, add_mask: bool) -> Tensor:
    temp = query.bmm(key.transpose(1, 2))
    with torch.no_grad():
        if add_mask:
            mask = generate_square_subsequent_mask(temp.shape[2])
            temp += mask
    scale = query.size(-1) ** 0.5
    scaled_temp = temp / scale
    softmax_scaled_temp = F.softmax(scaled_temp, dim=-1)
    return softmax_scaled_temp.bmm(value)


def generate_square_subsequent_mask(size: int):
    """Generate a triangular (size, size) mask. From PyTorch docs."""
    with torch.no_grad():
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class Transformer(nn.Module):
    def __init__(
            self,
            num_encoder_layers: int = 1,
            num_decoder_layers: int = 1,
            embedding_dim: int = 1,  # has to be dividable by 2 because sin and cos have to be applied
            num_heads: int = 1,
            dim_feedforward: int = 7,
            dropout: float = 0.1,
            max_output_length: int = 5,
            max_sequence_length: int = 20,
            vocab_size: int = 6,
            lin_in_layer_output_dim: int = 4
    ):
        super().__init__()

        with torch.no_grad():
            self.encoder_embedding = torch.nn.Embedding(max_sequence_length, embedding_dim)
            self.decoder_embedding = torch.nn.Embedding(max_sequence_length, embedding_dim)

        self.embedding_dim = embedding_dim
        self.max_output_length = max_output_length

        ### encoder ###
        self.encoder_dropout_1 = nn.Dropout(dropout)
        self.encoder_dropout_2 = nn.Dropout(dropout)

        # TODO a loop to create more than one head
        dim_q = dim_k = max(lin_in_layer_output_dim // num_heads, 1)

        self.encoder_multi_head_attention = MultiHeadAttention(num_heads, embedding_dim, dim_q, dim_k, add_mask=False)
        self.encoder_fc_layer = nn.Sequential(nn.Linear(embedding_dim, dim_feedforward),
                                              nn.ReLU(),
                                              nn.Linear(dim_feedforward, embedding_dim))

        self.norm = nn.LayerNorm(embedding_dim)

        ### decoder ###
        self.decoder_dropout_1 = nn.Dropout(dropout)
        self.decoder_dropout_2 = nn.Dropout(dropout)
        self.decoder_dropout_3 = nn.Dropout(dropout)

        self.decoder_multi_head_attention_1 = MultiHeadAttention(num_heads, embedding_dim, dim_q, dim_k, add_mask=True)
        self.decoder_multi_head_attention_2 = MultiHeadAttention(num_heads, embedding_dim, dim_q, dim_k, add_mask=False)
        self.decoder_fc_layer = nn.Sequential(nn.Linear(embedding_dim, dim_feedforward),
                                              nn.ReLU(),
                                              nn.Linear(dim_feedforward, embedding_dim))

        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward_encoder(self, src: Tensor):
        # create embedding
        embedded_src = self.encoder_embedding(src) * math.sqrt(self.embedding_dim)
        seq_len, embedding_dimension = embedded_src.size(1), embedded_src.size(2)

        # add positional encoding
        pos_encoding = position_encoding(seq_len, embedding_dimension).permute(1, 0, 2)  # -> (1, seq_len, embedding_dim)
        embedded_src += pos_encoding
        positional_embedded_source = embedded_src

        # self attention head
        # TODO make a loop to have more than one head
        query = positional_embedded_source
        key = positional_embedded_source
        value = positional_embedded_source

        encoder_multi_head_attention_output = self.encoder_multi_head_attention(query, key, value)
        encoder_multi_head_attention_output = self.encoder_dropout_1(encoder_multi_head_attention_output)
        # TODO end of loop

        # residual connection around the self attention head
        encoder_att_head_residual_connection = query + encoder_multi_head_attention_output
        decoder_self_attention_output = self.norm(encoder_att_head_residual_connection)

        # fully connected layer
        encoder_fc_output = self.encoder_fc_layer(decoder_self_attention_output)
        encoder_fc_output = self.encoder_dropout_2(encoder_fc_output)

        # residual connection around the fully connected layer
        encoder_fc_res_connection = decoder_self_attention_output + encoder_fc_output
        encoder_output = self.norm(encoder_fc_res_connection)

        return encoder_output

    def forward_decoder(self, encoder_output: Tensor, tgt: Tensor):
        embedded_tgt = self.decoder_embedding(tgt) * math.sqrt(self.embedding_dim)
        seq_len, embedding_dimension = embedded_tgt.size(1), embedded_tgt.size(2)

        pos_encoding = position_encoding(seq_len, embedding_dimension).permute(1, 0, 2)
        embedded_tgt += pos_encoding
        positional_embedded_tgt = embedded_tgt

        # TODO loop
        tgt_query = positional_embedded_tgt
        tgt_key = positional_embedded_tgt
        tgt_value = positional_embedded_tgt

        decoder_self_attention_tgt = self.decoder_multi_head_attention_1(tgt_query, tgt_key, tgt_value)
        decoder_self_attention_tgt = self.decoder_dropout_1(decoder_self_attention_tgt)
        decoder_att_head_residual_connection = tgt_query + decoder_self_attention_tgt
        decoder_self_attention_output = self.norm(decoder_att_head_residual_connection)

        decoder_attention_2 = self.decoder_multi_head_attention_2(decoder_self_attention_output, encoder_output, encoder_output)
        decoder_attention_2 = self.decoder_dropout_2(decoder_attention_2)
        decoder_attention_2_head_residual_connection = decoder_self_attention_output + decoder_attention_2
        decoder_attention_2_output = self.norm(decoder_attention_2_head_residual_connection)

        # fully connected layer
        decoder_fc_output = self.decoder_fc_layer(decoder_self_attention_output)
        decoder_fc_output = self.decoder_dropout_3(decoder_fc_output)

        # residual connection around the fully connected layer
        decoder_fc_res_connection = decoder_attention_2_output + decoder_fc_output
        decoder_output_fc_output = self.norm(decoder_fc_res_connection)
        # TODO end loop

        lin_output = self.linear(decoder_output_fc_output)

        decoder_output = torch.softmax(lin_output, dim=-1)
        return decoder_output

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        ### encoder ###
        encoder_output = self.forward_encoder(src)

        ### decoder ###
        decoder_output = self.forward_decoder(encoder_output=encoder_output, tgt=tgt)

        return decoder_output

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method to use at inference time. Predict y from x one token at a time. This method is greedy
        decoding. Beam search can be used instead for a potential accuracy boost.

        Input
            x: (B, Sx) with elements in (0, C) where C is num_classes
        Output
            (B, C, Sy) logits
        """
        encoded_x = self.forward_encoder(x)

        output_tokens = (torch.ones((x.shape[0], self.max_output_length))).type_as(x).long() # (B, max_length)
        output_tokens[:, 0] = 0  # Set start token
        for Sy in range(1, self.max_output_length):
            y = output_tokens[:, :Sy]  # (B, Sy)
            output = self.forward_decoder(encoder_output=encoded_x, tgt=y)  # (Sy, B, C)
            output = torch.argmax(output, dim=-1)  # (Sy, B)
            output_tokens[:, Sy] = output[:, -1]  # Set the last output token
        return output_tokens


def position_encoding(seq_len: int, embedding_dim: int):
    """
    returns positional encoding for seq_len, 1, embedding_dim
    """
    with torch.no_grad():
        pos_encoding = torch.zeros(seq_len, embedding_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
    return pos_encoding

# remove randomness
torch.manual_seed(42)


### CREATE DATA ###

BATCH_SIZE = 2
TRAIN_FRAC = 0.5
N = 4
S = 8  # target sequence length. input sequence will be twice as long
vocab_size = 6  # number of "classes", including 0, the "start token", and 1, the "end token"

X = torch.tensor([[0, 2, 2, 3, 3, 4, 4, 5, 5, 1],
                  [0, 2, 2, 3, 3, 4, 4, 5, 5, 1]])

Y = torch.tensor([[0, 2, 3, 4, 5, 1],
                  [0, 2, 3, 4, 5, 1]])

dataset = list(zip(X, Y))  # This fulfills the pytorch.utils.data.Dataset interface
dataloader_train = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

### training ###
max_sequence_length = 20
embedding_dim = 4  # has to be dividable by 2 because of son/cos positional encoding

model = Transformer(max_output_length=Y.shape[1],
                    vocab_size=vocab_size,
                    max_sequence_length=max_sequence_length,
                    embedding_dim=embedding_dim
                    )

model.train()
learning_rate = 0.1
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_loss = 0
loss = nn.CrossEntropyLoss()

for step in range(20):
    x, y = next(iter(dataloader_train))
    target = y[:, :-1]
    loss_target = y[:, 1:]

    optimizer.zero_grad()
    logits = model(x, target)
    logits_argmax = torch.argmax(logits, dim=-1).type(torch.float)

    int_64_loss_target = loss_target.type(torch.int64)
    one_hot_loss_targets = F.one_hot(int_64_loss_target, num_classes=vocab_size).type(torch.float)

    loss_val = loss(logits, one_hot_loss_targets)
    print(f'loss_val: {loss_val}')
    print(f'logits: {logits_argmax}')
    print(f'target: {target}')

    loss_val.backward()
    optimizer.step()

### prediction (cheating because we use the training data) ###

model.eval()
x, y = next(iter(dataloader_train))
print(x[0:1])
print(model.predict(x[0:1]))