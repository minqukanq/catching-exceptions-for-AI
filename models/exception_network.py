import torch
import torch.nn as nn
from models.self_attention import SelfAttention
from models.encoder_block import EncoderBlock

class ExceptionLSTMNet(nn.Module):
    def __init__(self, n_classes, encoder, embedding_size, lstm_hidden_size, fc_hidden_size, drop_out):
        super(ExceptionLSTMNet, self).__init__()

        self.embedding = encoder
        for param in self.embedding.parameters():
            param.requires_grad = False

#         embedding_weights = torch.FloatTensor(np.random.uniform(-1,1, size=(vocab_size, embedding_size)))
#         self.embedding = nn.Embedding(vocab_size, embedding_size, _weight=embedding_weights)
        self.bi_lstm = nn.LSTM(input_size=embedding_size, hidden_size=lstm_hidden_size, num_layers=2,
                                batch_first=True, bidirectional=True, dropout=drop_out)
        self.dropout = nn.Dropout(drop_out)
        self.linear = nn.Linear(lstm_hidden_size * 2, fc_hidden_size)
        self.classifier = nn.Linear(fc_hidden_size, n_classes)
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask):
        # Embedding Layer
#         embedded_sentence = self.embedding(input_x)
        embedded_sentence = self.embedding(input_ids, attention_mask)[0]
    
        # Bi-LSTM Layer
        lstm_out, _ = self.bi_lstm(embedded_sentence)
        lstm_out_pool = torch.mean(lstm_out, dim=1)
        
        output = self.linear(lstm_out)
        output = self.relu(output)
        output = self.dropout(output)
        
        output = self.classifier(output.transpose(0,1)[-1])
        return output


class ExceptionAttentionNet(nn.Module):
    def __init__(self, n_classes, encoder, embedding_size, heads, num_layers, forward_expansion, drop_out):
        super(ExceptionAttentionNet, self).__init__()

        self.embedding = encoder
        for param in self.embedding.parameters():
            param.requires_grad = False

        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embedding_size,
                    heads,
                    dropout=drop_out,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.classifier = nn.Linear(embedding_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        # Embedding Layer
        out = self.embedding(input_ids, attention_mask)[0]

        for layer in self.layers:
            out = layer(out,out,out)

        out = self.classifier(out.transpose(0,1)[-1])
        return out