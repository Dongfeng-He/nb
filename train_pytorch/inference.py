import os
import pandas as pd
from evaluation import *
import random
import copy
from keras.preprocessing import text, sequence
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
import numpy as np
import time
import gc


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix):
        super(NeuralNet, self).__init__()
        unique_word_num = embedding_matrix.shape[0]
        embed_size = embedding_matrix.shape[1]
        lstm_size = 128
        dense_size = 512
        # 嵌入层
        self.embedding = nn.Embedding(unique_word_num, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)
        # LSTM
        self.lstm1 = nn.LSTM(embed_size, lstm_size, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_size * 2, lstm_size, bidirectional=True, batch_first=True)
        # 全连接层
        self.linear1 = nn.Linear(dense_size, dense_size)
        self.linear2 = nn.Linear(dense_size, dense_size)
        # 输出层
        self.linear_out = nn.Linear(dense_size, 1)
        self.linear_aux_out = nn.Linear(dense_size, 5)

    def forward(self, x):
        # 嵌入层
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)
        # LSTM
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        # pooling
        avg_pool = torch.mean(h_lstm2, 1)
        max_pool, _ = torch.max(h_lstm2, 1)
        # 全连接层
        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1 = F.relu(self.linear1(h_conc))
        h_conc_linear2 = F.relu(self.linear2(h_conc))
        # 拼接
        hidden = h_conc + h_conc_linear1 + h_conc_linear2
        # 输出层，用 sigmoid 就用 BCELoss，不用 sigmoid 就用 BCEWithLogitsLoss
        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)
        return out


class Trainer:
    def __init__(self, data_dir, batch_size=512, part=1., debug_mode=False):
        self.data_dir = data_dir
        self.debug_mode = debug_mode
        self.identity_list = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
        self.toxicity_type_list = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
        self.stopwords = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
        self.seed_everything()
        self.max_len = 220
        self.batch_size = batch_size
        self.split_ratio = 0.95
        self.sample_num = 1804874
        self.train_df = pd.read_csv(os.path.join(self.data_dir, "train.csv")).head(int(self.sample_num * part))
        self.test_df = pd.read_csv(os.path.join(self.data_dir, "test.csv"))
        self.train_len = int(len(self.train_df) * self.split_ratio)
        self.evaluator = self.init_evaluator()

    def seed_everything(self, seed=1234):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def init_evaluator(self):
        # 初始化评分函数类
        y_true = self.train_df['target'].values
        y_identity = self.train_df[self.identity_list].values
        valid_y_true = y_true[self.train_len:]
        valid_y_identity = y_identity[self.train_len:]
        evaluator = JigsawEvaluator(valid_y_true, valid_y_identity) # y_true 必须是0或1，不能是离散值
        return evaluator

    def create_dataloader(self):
        # 读取输入输出
        train_comments = self.train_df["comment_text"].astype(str)
        train_label = self.train_df["target"].values
        train_type_labels = self.train_df[self.toxicity_type_list].values
        # tokenizer 训练
        test_comments = self.test_df["comment_text"].astype(str)
        tokenizer = text.Tokenizer(filters=self.stopwords)
        tokenizer.fit_on_texts(list(train_comments)[self.train_len:])    # train_comments 是 dataframe 的一列，是 Series 类， list(train_comments) 直接变成 list
        # tokenization
        train_tokens = tokenizer.texts_to_sequences(train_comments)     # 可以给 Series 也可以给 list？
        test_tokens = tokenizer.texts_to_sequences(test_comments)
        # 用 sequence 类补到定长
        train_tokens = sequence.pad_sequences(train_tokens, maxlen=self.max_len)
        test_tokens = sequence.pad_sequences(test_tokens, maxlen=self.max_len)
        # 划分训练集和验证集
        valid_tokens = train_tokens[self.train_len:]
        valid_label = train_label[self.train_len:]
        valid_type_labels = train_type_labels[self.train_len:]
        train_tokens = train_tokens[:self.train_len]
        train_label = train_label[:self.train_len]
        train_type_labels = train_type_labels[:self.train_len]
        # 将符号化数据转成 tensor
        train_x_tensor = torch.tensor(train_tokens, dtype=torch.long)
        valid_x_tensor = torch.tensor(valid_tokens, dtype=torch.long)
        train_y_tensor = torch.tensor(np.hstack([train_label[:, np.newaxis], train_type_labels]), dtype=torch.float32)
        valid_y_tensor = torch.tensor(np.hstack([valid_label[:, np.newaxis], valid_type_labels]), dtype=torch.float32)
        if torch.cuda.is_available():
            train_x_tensor = train_x_tensor.cuda()
            valid_x_tensor = valid_x_tensor.cuda()
            train_y_tensor = train_y_tensor.cuda()
            valid_y_tensor = valid_y_tensor.cuda()
        # 将 tensor 转成 dataset，训练数据和标签一一对应，用 dataloader 加载的时候 dataset[:-1] 是 x，dataset[-1] 是 y
        train_dataset = data.TensorDataset(train_x_tensor, train_y_tensor)
        valid_dataset = data.TensorDataset(valid_x_tensor, valid_y_tensor)
        # 将 dataset 转成 dataloader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        # 返回训练数据
        return train_loader, valid_loader, tokenizer

    def create_emb_weights(self, word_index):
        with open(os.path.join(self.data_dir, "crawl-300d-2M.vec"), "r") as f:
            fasttext_emb_dict = {}
            for i, line in enumerate(f):
                if i == 1000 and self.debug_mode: break
                split = line.strip().split(" ")
                word = split[0]
                if word not in word_index: continue
                emb = np.array([float(num) for num in split[1:]])
                fasttext_emb_dict[word] = emb
        with open(os.path.join(self.data_dir, "glove.840B.300d.txt"), "r") as f:
            glove_emb_dict = {}
            for i, line in enumerate(f):
                if i == 1000 and self.debug_mode: break
                split = line.strip().split(" ")
                word = split[0]
                if word not in word_index: continue
                emb = np.array([float(num) for num in split[1:]])
                glove_emb_dict[word] = emb
        word_embedding = np.zeros((len(word_index) + 1, 600))     # tokenizer 自动留出0用来 padding
        for word, index in word_index.items():
            # 如果找不到 emb，尝试小写或首字母大写
            if word not in fasttext_emb_dict and word not in glove_emb_dict:
                word = word.lower()
                if word not in fasttext_emb_dict and word not in glove_emb_dict:
                    word = word.title()
                    if word not in fasttext_emb_dict and word not in glove_emb_dict:
                        word = word.upper()
            fasttext_emb = fasttext_emb_dict[word] if word in fasttext_emb_dict else np.random.uniform(-0.25, 0.25, 300)
            glove_emb = glove_emb_dict[word] if word in glove_emb_dict else np.random.uniform(-0.25, 0.25, 300)
            word_embedding[index] = np.concatenate((fasttext_emb, glove_emb), axis=-1)
        return np.array(word_embedding)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def custom_loss(self, y_pred, y_batch, target_weight=None, aux_weight=None, identity_weight=None):
        target_pred = y_pred[:, 0]
        target_true = y_batch[:, 0]
        aux_pred = y_pred[:, 1:]
        aux_true = y_batch[:, 1:]
        target_loss = nn.BCEWithLogitsLoss(reduction="none")(target_pred, target_true)
        target_loss = torch.mean(target_loss * target_weight)
        if True:
            aux_loss = nn.BCEWithLogitsLoss()(aux_pred, aux_true)
        else:
            aux_loss = nn.BCEWithLogitsLoss(reduction="none")(aux_pred, aux_true)
            aux_loss = torch.mean(aux_loss * aux_weight)
        return target_loss, aux_loss

    def eval(self):
        if self.debug_mode: self.epochs = 1
        # 加载 dataloader
        train_loader, valid_loader, tokenizer = self.create_dataloader()
        # 生成 embedding
        word_embedding = self.create_emb_weights(tokenizer.word_index)
        self.seed_everything(1234)
        model = NeuralNet(word_embedding)
        # 读取模型，加载词向量
        temp_dict = torch.load("/Users/hedongfeng/Desktop/model.model")
        temp_dict['embedding.weight'] = torch.tensor(word_embedding)
        model.load_state_dict(temp_dict)

        if torch.cuda.is_available():
            model.cuda()
        # 计算验证集
        model.eval()
        y_pred = np.zeros((len(self.train_df) - self.train_len))
        for i, batch_data in enumerate(valid_loader):
            x_batch = batch_data[:-1]
            y_batch = batch_data[-1]
            batch_y_pred = self.sigmoid(model(*x_batch).detach().cpu().numpy())[:, 0]
            y_pred[i * self.batch_size: (i + 1) * self.batch_size] = batch_y_pred
        # 计算得分
        auc_score = self.evaluator.get_final_metric(y_pred)
        print(auc_score)


if __name__ == "__main__":
    data_dir = "/Users/hedongfeng/PycharmProjects/unintended_bias/data/"
    trainer = Trainer(data_dir, debug_mode=False)
    trainer.eval()
