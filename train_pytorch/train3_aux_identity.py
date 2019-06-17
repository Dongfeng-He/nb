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
import math
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
        self.linear_identity_out = nn.Linear(dense_size, 9)

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
        identity_result = self.linear_identity_out(hidden)
        out = torch.cat([result, aux_result, identity_result], 1)
        return out


class Trainer:
    def __init__(self, data_dir, model_name, epochs=5, batch_size=512, part=1., debug_mode=False):
        self.data_dir = data_dir
        self.debug_mode = debug_mode
        self.model_name = model_name
        self.identity_list = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
        self.toxicity_type_list = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
        self.weight_dict = {"severe_toxicity": 1000, "obscene": 195, "identity_attack": 277, "insult": 21,
                            "threat": 608, "male": 44, "female": 32, "homosexual_gay_or_lesbian": 197, "christian": 47,
                            "jewish": 242, "muslim": 132, "black": 130, "white": 89, "psychiatric_or_mental_illness": 368,
                            "np": 12, "pn": 15}
        self.stopwords = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
        self.seed_everything()
        self.seed = 5
        self.max_len = 220
        self.epochs = epochs
        self.batch_size = batch_size
        self.split_ratio = 0.95
        self.sample_num = 1804874
        if not self.debug_mode:
            self.train_df = pd.read_csv(os.path.join(self.data_dir, "predict.csv")).head(int(self.sample_num * part)).fillna(0.)
            self.test_df = pd.read_csv(os.path.join(self.data_dir, "test.csv"))
        else:
            self.train_df = pd.read_csv(os.path.join(self.data_dir, "predict.csv")).head(1000).fillna(0.)
            self.test_df = pd.read_csv(os.path.join(self.data_dir, "test.csv")).head(1000)
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

        # 身份原始值
        train_identity_values = self.train_df[self.identity_list].fillna(0.).values
        # 所有身份原始值之和
        train_identity_sum = train_identity_values.sum(axis=1)
        # 将身份之和限制在1以下（sigmoid）
        train_identity_sum_label = np.where(train_identity_sum > 1, 1, train_identity_sum)
        # 身份01值
        train_identity_binary = copy.deepcopy(self.train_df[self.identity_list])
        for column in self.identity_list:
            train_identity_binary[column] = np.where(train_identity_binary[column] > 0.5, 1, 0)
        # 身份01值有一个就算1
        train_identity_binary_sum = train_identity_binary.sum(axis=1)
        train_identity_or_binary = np.where(train_identity_binary_sum >= 1, 1, 0)
        # 所有身份标签
        train_identity_type_labels = train_identity_values
        train_identity_type_binary_lables = train_identity_binary
        train_identity_sum_label = train_identity_sum_label
        train_identity_binary_label = train_identity_or_binary

        # tokenizer 训练
        test_comments = self.test_df["comment_text"].astype(str)
        tokenizer = text.Tokenizer(filters=self.stopwords)
        tokenizer.fit_on_texts(list(train_comments) + list(test_comments))    # train_comments 是 dataframe 的一列，是 Series 类， list(train_comments) 直接变成 list
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
        valid_identity_type_labels = train_identity_type_labels[self.train_len:]
        train_identity_type_labels = train_identity_type_labels[:self.train_len]
        valid_identity_type_binary_lables = train_identity_type_binary_lables[self.train_len:]
        train_identity_type_binary_lables = train_identity_type_binary_lables[:self.train_len]
        valid_identity_sum_label = train_identity_sum_label[self.train_len:]
        train_identity_sum_label = train_identity_sum_label[:self.train_len]
        valid_identity_binary_label = train_identity_binary_label[self.train_len:]
        train_identity_binary_label = train_identity_binary_label[:self.train_len]

        # 计算样本权重
        target_weight, aux_weight, identity_weight = self.cal_sample_weights()

        # 将符号化数据转成 tensor
        train_x_tensor = torch.tensor(train_tokens, dtype=torch.long)
        valid_x_tensor = torch.tensor(valid_tokens, dtype=torch.long)
        train_y_tensor = torch.tensor(np.hstack([train_label[:, np.newaxis], train_type_labels, train_identity_type_labels]), dtype=torch.float32)
        valid_y_tensor = torch.tensor(np.hstack([valid_label[:, np.newaxis], valid_type_labels, valid_identity_type_labels]), dtype=torch.float32)
        target_weight_tensor = torch.tensor(target_weight, dtype=torch.float32)
        aux_weight_tensor = torch.tensor(aux_weight, dtype=torch.float32)
        identity_weight_tensor = torch.tensor(identity_weight, dtype=torch.float32)
        if torch.cuda.is_available():
            train_x_tensor = train_x_tensor.cuda()
            valid_x_tensor = valid_x_tensor.cuda()
            train_y_tensor = train_y_tensor.cuda()
            valid_y_tensor = valid_y_tensor.cuda()
            target_weight_tensor = target_weight_tensor.cuda()
            aux_weight_tensor = aux_weight_tensor.cuda()
            identity_weight_tensor = identity_weight_tensor.cuda()
        # 将 tensor 转成 dataset，训练数据和标签一一对应，用 dataloader 加载的时候 dataset[:-1] 是 x，dataset[-1] 是 y
        train_dataset = data.TensorDataset(train_x_tensor, train_y_tensor, target_weight_tensor, aux_weight_tensor, identity_weight_tensor)
        valid_dataset = data.TensorDataset(valid_x_tensor, valid_y_tensor)
        # 将 dataset 转成 dataloader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        # 返回训练数据
        return train_loader, valid_loader, tokenizer

    def cal_sample_weights(self):
        # aux weight
        aux_weight = np.zeros((len(self.train_df), len(self.toxicity_type_list)))
        for i, column in enumerate(self.toxicity_type_list):
            weight = math.pow(self.weight_dict[column], 0.5)
            aux_weight[:, i] = np.where(self.train_df[column] > 0.5, weight, 1)
        # identity weight
        identity_weight = np.zeros((len(self.train_df), len(self.identity_list)))
        for i, column in enumerate(self.identity_list):
            weight = math.pow(self.weight_dict[column], 0.5)
            identity_weight[:, i] = np.where(self.train_df[column] > 0.5, weight, 1)
        # target weight
        for column in self.identity_list + ["target"]:
            self.train_df[column] = np.where(self.train_df[column] > 0.5, True, False)
        target_weight = np.ones(len(self.train_df))
        target_weight += self.train_df["target"]
        if False:
            target_weight += (~self.train_df["target"]) * self.train_df[self.identity_list].sum(axis=1)
            target_weight += self.train_df["target"] * (~self.train_df[self.identity_list]).sum(axis=1) * 5
        else:
            target_weight += (~self.train_df["target"]) * np.where(self.train_df[self.identity_list].sum(axis=1) > 0, 1, 0) * 3
            target_weight += self.train_df["target"] * np.where((~self.train_df[self.identity_list]).sum(axis=1) > 0, 1, 0) * 3
        target_weight /= target_weight.mean()
        # 只留训练集
        target_weight = target_weight[:self.train_len]
        aux_weight = aux_weight[:self.train_len, :]
        identity_weight = identity_weight[:self.train_len, :]
        return target_weight, aux_weight, identity_weight

    def create_emb_weights(self, word_index):
        # 构建词向量字典
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
        # 为训练集和测试集出现过的词构建词向量矩阵
        word_embedding = np.zeros((len(word_index) + 1, 600))     # tokenizer 自动留出0用来 padding
        np.random.seed(1234)
        fasttext_random_emb = np.random.uniform(-0.25, 0.25, 300)   # 用于 fasttext 找不到词语时
        np.random.seed(1235)
        glove_random_emb = np.random.uniform(-0.25, 0.25, 300)  # 用于 glove 找不到词语时
        for word, index in word_index.items():
            # 如果找不到 emb，尝试小写或首字母大写
            if word not in fasttext_emb_dict and word not in glove_emb_dict:
                word = word.lower()
                if word not in fasttext_emb_dict and word not in glove_emb_dict:
                    word = word.title()
                    if word not in fasttext_emb_dict and word not in glove_emb_dict:
                        word = word.upper()
            fasttext_emb = fasttext_emb_dict[word] if word in fasttext_emb_dict else fasttext_random_emb
            glove_emb = glove_emb_dict[word] if word in glove_emb_dict else glove_random_emb
            word_embedding[index] = np.concatenate((fasttext_emb, glove_emb), axis=-1)
        return np.array(word_embedding)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def custom_loss(self, y_pred, y_batch, target_weight=1., aux_weight=1., identity_weight=1.):
        target_pred = y_pred[:, 0]
        target_true = y_batch[:, 0]
        aux_pred = y_pred[:, 1: 6]
        aux_true = y_batch[:, 1: 6]
        identity_pred = y_pred[:, 6:]
        identity_batch = y_batch[:, 6:]
        target_loss = nn.BCEWithLogitsLoss(reduction="none")(target_pred, target_true)
        target_loss = torch.mean(target_loss * target_weight)
        aux_loss = nn.BCEWithLogitsLoss(reduction="none")(aux_pred, aux_true)
        aux_loss = torch.mean(aux_loss * aux_weight)
        identity_loss = nn.BCEWithLogitsLoss(reduction="none")(identity_pred, identity_batch)
        identity_loss = torch.mean(identity_loss * identity_weight)
        return target_loss, aux_loss, identity_loss

    def train(self):
        if self.debug_mode: self.epochs = 1
        # 加载 dataloader
        train_loader, valid_loader, tokenizer = self.create_dataloader()
        # 生成 embedding
        word_embedding = self.create_emb_weights(tokenizer.word_index)
        # 训练
        self.seed_everything(1234)
        model = NeuralNet(word_embedding)
        if torch.cuda.is_available():
            model.cuda()
        lr = 1e-3
        # param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()] # 可以为不同层设置不同的学习速率
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # 渐变学习速率
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)
        # 损失函数
        loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        # 训练
        previous_auc_score = 0
        stop_flag = 0
        for epoch in range(self.epochs):
            start_time = time.time()
            # 调整一次学习速率
            scheduler.step()
            # 切换为训练模式
            model.train()
            # 初始化当前 epoch 的 loss
            avg_loss = 0.
            # 加载每个 batch 并训练
            for batch_data in train_loader:
                x_batch = batch_data[0]
                y_batch = batch_data[1]
                target_weight_batch = batch_data[2]
                aux_weight_batch = batch_data[3]
                identity_weight_batch = batch_data[4]
                #y_pred = model(*x_batch)
                y_pred = model(x_batch)
                target_loss, aux_loss, identity_loss = self.custom_loss(y_pred, y_batch, target_weight_batch, aux_weight_batch, identity_weight_batch)
                loss = target_loss + aux_loss + identity_loss
                #loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() / len(train_loader)
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
            if auc_score < previous_auc_score:
                if stop_flag == 0:
                    stop_flag += 1
                else:
                    break
            else:
                stop_flag = 0
                previous_auc_score = auc_score
            print("epoch: %d duration: %d min auc_score: %.4f" % (epoch, int((time.time() - start_time) / 60), auc_score))
            if not self.debug_mode and epoch > 0:
                temp_dict = model.state_dict()
                del temp_dict['embedding.weight']
                torch.save(temp_dict, os.path.join(self.data_dir, "model/model[pytorch][%s]_%d_%.5f" % (self.model_name, epoch, auc_score)))
        # del 训练相关输入和模型
        training_history = [train_loader, valid_loader, tokenizer, word_embedding, model, optimizer, scheduler]
        for variable in training_history:
            del variable
        gc.collect()


if __name__ == "__main__":
    data_dir = "/Users/hedongfeng/PycharmProjects/unintended_bias/data/"
    trainer = Trainer(data_dir, "model_name", debug_mode=True)
    trainer.train()
