import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from transformers import AutoModel

class CEFE(nn.Module):
    def __init__(self, args):
        super(CEFE, self).__init__()

        self.pos_pro = pos_process(args)
        self.course_pred = error_pred(args, embed_dim=args.att_hid_dim + args.bert_dim)
        self.fine_pred = nn.ModuleList([
            error_pred(args, args.att_hid_dim + args.bert_dim) for _ in range(4)
        ])
        self.embedding = nn.Embedding(num_embeddings=45, embedding_dim=args.embed_dim)
        self.bert = AutoModel.from_pretrained(args.model_path)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.relu = nn.ReLU()


    def forward(self, encoded_inputs, sent_pos):
        pos_embedding = self.embedding(sent_pos) # [batch_size, max_len, 50]
        pos_embedding = self.pos_pro(pos_embedding) # [batch_size, 50]

        bert_embedding = self.bert(**encoded_inputs).last_hidden_state[:, 0, :]
        embedding = torch.cat([pos_embedding, bert_embedding], dim=1)

        courseGrainedErrorPred, hidden = self.course_pred(embedding)

        fineGrainedErrorPred = []
        for fine_pred_layer in self.fine_pred:
            fine_pred, _ = fine_pred_layer(embedding, hidden)
            fineGrainedErrorPred.append(fine_pred)
        fineGrainedErrorPred = torch.cat(fineGrainedErrorPred, dim=1)
        return courseGrainedErrorPred, fineGrainedErrorPred


class CEFE_pos(nn.Module):
    def __init__(self, args):
        super(CEFE_pos, self).__init__()

        self.pos_pro = pos_process(args)
        self.course_pred = error_pred(args, embed_dim=args.att_hid_dim)
        self.fine_pred = nn.ModuleList([
            error_pred(args, embed_dim=args.att_hid_dim) for _ in range(4)
        ])
        self.embedding = nn.Embedding(num_embeddings=45, embedding_dim=args.embed_dim)
        self.bert = AutoModel.from_pretrained(args.model_path)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.relu = nn.ReLU()


    def forward(self, encoded_inputs, sent_pos):
        pos_embedding = self.embedding(sent_pos) # [batch_size, max_len, 50]
        pos_embedding = self.pos_pro(pos_embedding) # [batch_size, 50]

        courseGrainedErrorPred, hidden = self.course_pred(pos_embedding)

        fineGrainedErrorPred = []
        for fine_pred_layer in self.fine_pred:
            fine_pred, _ = fine_pred_layer(pos_embedding, hidden)
            fineGrainedErrorPred.append(fine_pred)
        fineGrainedErrorPred = torch.cat(fineGrainedErrorPred, dim=1)
        return courseGrainedErrorPred, fineGrainedErrorPred


class CEFE_bert(nn.Module):
    def __init__(self, args):
        super(CEFE_bert, self).__init__()

        self.pos_pro = pos_process(args)
        self.course_pred = error_pred(args, embed_dim=args.bert_dim)
        self.fine_pred = nn.ModuleList([
            error_pred(args, embed_dim=args.bert_dim) for _ in range(4)
        ])
        self.embedding = nn.Embedding(num_embeddings=45, embedding_dim=args.embed_dim)
        self.bert = AutoModel.from_pretrained(args.model_path)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.relu = nn.ReLU()


    def forward(self, encoded_inputs, sent_pos):
        bert_embedding = self.bert(**encoded_inputs).last_hidden_state[:, 0, :]

        courseGrainedErrorPred, hidden = self.course_pred(bert_embedding)

        fineGrainedErrorPred = []
        for fine_pred_layer in self.fine_pred:
            fine_pred, _ = fine_pred_layer(bert_embedding, hidden)
            fineGrainedErrorPred.append(fine_pred)
        fineGrainedErrorPred = torch.cat(fineGrainedErrorPred, dim=1)
        return courseGrainedErrorPred, fineGrainedErrorPred


class error_pred(nn.Module):
    def __init__(self, args, embed_dim=32+768):
        super(error_pred, self).__init__()
        self.args = args

        self.dense = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=args.dense_hid_dim),
            nn.ReLU()
        )
        self.dense_2 = nn.Sequential(
            nn.Linear(in_features=args.dense_hid_dim, out_features=4),
            nn.Sigmoid()
        )
        self.dense_3 = nn.Sequential(
            nn.Linear(in_features=args.dense_hid_dim * 2, out_features=4),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        # 对 dense 中的 Linear 层进行初始化
        for layer in self.dense:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, course_hid=None):
        hidden = self.dense(x)
        if course_hid is not None:
            hidden = torch.cat((hidden, course_hid), dim=1)
            x = self.dense_3(hidden)
        else:
            x = self.dense_2(hidden)
        return x, hidden
    

class pos_process(nn.Module):
    def __init__(self, args):
        super(pos_process, self).__init__()
        self.cnn = nn.Conv1d(in_channels=args.embed_dim, out_channels=args.conv_out_channels, kernel_size=args.kernal_size)
        self.lstm = nn.LSTM(input_size=args.conv_out_channels, hidden_size=args.lstm_hid_dim)
        self.att = AttentionPooling(input_dim=args.lstm_hid_dim, hidden_dim=args.att_hid_dim)

        self.reset_parameters()

    def reset_parameters(self):
        # 初始化 CNN
        nn.init.kaiming_uniform_(self.cnn.weight, nonlinearity='relu')
        nn.init.zeros_(self.cnn.bias)

        # 初始化 LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, pos):
        pos_conv = self.cnn(pos.permute(0, 2, 1))
        pos_conv = pos_conv.permute(0, 2, 1)
        pos, _ = self.lstm(pos_conv)
        pos = self.att(pos)
        return pos
    

class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)  # 投影层
        self.v = nn.Parameter(torch.randn(hidden_dim))  # 可学习的注意力向量

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        nn.init.normal_(self.v, mean=0.0, std=0.1)

    def forward(self, x, mask=None):
        # x: [dim_1, dim_2, dim_3]
        x_proj = torch.tanh(self.proj(x))  # activative func [dim_1, dim_2, dim_3]
        scores = torch.matmul(x_proj, self.v)  # att score [dim_1, dim_2]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('1e-9'))
        weights = F.softmax(scores, dim=1)    # softmax [dim_1, dim_2]

        # [dim_1, dim_3]
        outputs = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return outputs
    

class Attention(nn.Module):
    def __init__(self, input_dim, output_dim, isMultiHead, num_heads=8):
        super(Attention, self).__init__()
        self.isMultiHead = isMultiHead
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.num_heads = num_heads
        if isMultiHead:
            assert output_dim % num_heads == 0

        self.projection_dim = output_dim // num_heads
        self.wq = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.wk = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.wv = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.dense = nn.Linear(in_features=output_dim, out_features=output_dim)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.wq, self.wk, self.wv, self.dense]:
            init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                init.zeros_(layer.bias)

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        matmul_qk = torch.matmul(query, key.transpose(-2, -1)) # q*k
        logits = matmul_qk / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32)) # q*k / sqrt(d)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-1e9')) # mask on
        attention_weights = nn.functional.softmax(logits, dim=-1) # softmax
        output = torch.matmul(attention_weights, value) # ()*v
        return output, attention_weights

    def split_heads(self, x, batch_size):
        x = torch.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return x

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        query = self.wq(q)
        key = self.wk(k)
        value = self.wv(v)
        if self.isMultiHead:
            query = self.split_heads(query, batch_size)
            key = self.split_heads(key, batch_size)
            value = self.split_heads(value, batch_size)

        attention, _ = self.scaled_dot_product_attention(query, key, value, mask)
        if self.isMultiHead:
            attention = attention.permute(0, 2, 1, 3)
            attention = torch.reshape(attention, (batch_size, -1, self.output_dim))
        output = self.dense(attention)
        return output