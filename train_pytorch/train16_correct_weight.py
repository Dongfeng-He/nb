import os
import pandas as pd
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
from sklearn.metrics import roc_auc_score


class JigsawEvaluator:
    def __init__(self, y_true, y_identity, power=-5, overall_model_weight=0.25):
        self.y = (y_true >= 0.5).astype(int)
        self.y_i = (y_identity >= 0.5).astype(int)
        self.n_subgroups = self.y_i.shape[1]
        self.power = power
        self.overall_model_weight = overall_model_weight

    @staticmethod
    def _compute_auc(y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return np.nan

    def _compute_subgroup_auc(self, i, y_pred):
        mask = self.y_i[:, i] == 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def _compute_bpsn_auc(self, i, y_pred):
        mask = self.y_i[:, i] + self.y == 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def _compute_bnsp_auc(self, i, y_pred):
        mask = self.y_i[:, i] + self.y != 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def compute_bias_metrics_for_model(self, y_pred):
        records = np.zeros((3, self.n_subgroups))
        for i in range(self.n_subgroups):
            records[0, i] = self._compute_subgroup_auc(i, y_pred)
            records[1, i] = self._compute_bpsn_auc(i, y_pred)
            records[2, i] = self._compute_bnsp_auc(i, y_pred)
        return records

    def _calculate_overall_auc(self, y_pred):
        return roc_auc_score(self.y, y_pred)

    def _power_mean(self, array):
        total = sum(np.power(array, self.power))
        return np.power(total / len(array), 1 / self.power)

    def get_final_metric(self, y_pred):
        bias_metrics = self.compute_bias_metrics_for_model(y_pred)
        bias_score = np.average([
            self._power_mean(bias_metrics[0]),
            self._power_mean(bias_metrics[1]),
            self._power_mean(bias_metrics[2])
        ])
        overall_score = self.overall_model_weight * self._calculate_overall_auc(y_pred)
        bias_score = (1 - self.overall_model_weight) * bias_score
        return overall_score + bias_score


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            bce_loss = nn.BCEWithLogitsLoss(reduction="none")(inputs, targets)
        else:
            bce_loss = nn.BCELoss(reduction="none")(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        #focal_loss = (1 - pt) ** self.gamma * bce_loss
        if self.reduce:
            return torch.mean(focal_loss)
        else:
            return focal_loss


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
        self.linear3 = nn.Linear(dense_size * 2, dense_size)
        # 输出层
        self.linear_out = nn.Linear(dense_size, 1)
        self.linear_aux_out = nn.Linear(dense_size, 5)
        self.linear_identity_out = nn.Linear(dense_size, 9)
        self.linear_identity_out2 = nn.Linear(dense_size, dense_size)
        self.bn1 = nn.BatchNorm1d(dense_size)
        self.bn2 = nn.BatchNorm1d(dense_size)

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

        identity_hidden = self.linear_identity_out2(h_conc)
        identity_hidden = F.relu(identity_hidden)
        #identity_hidden = self.bn1(identity_hidden)
        identity_hidden = F.dropout(identity_hidden, p=0.3)
        identity_result = self.linear_identity_out(identity_hidden)
        h_conc2 = torch.cat((h_conc, identity_hidden), 1)
        gate_hidden = self.linear3(h_conc2)
        #gate_hidden = self.bn2(gate_hidden)
        gate = torch.sigmoid(gate_hidden)
        #gate = F.dropout(gate, p=0.3)
        h_conc = h_conc * gate

        h_conc_linear1 = F.relu(self.linear1(h_conc))
        h_conc_linear2 = F.relu(self.linear2(h_conc))
        # 拼接
        hidden = h_conc + h_conc_linear1 + h_conc_linear2
        # 输出层，用 sigmoid 就用 BCELoss，不用 sigmoid 就用 BCEWithLogitsLoss
        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result, identity_result], 1)
        return out


class Trainer:
    def __init__(self, model_name, epochs=5, batch_size=512, part=1., seed=1234, debug_mode=False):
        self.debug_mode = debug_mode
        self.model_name = model_name
        self.seed = seed
        self.identity_list = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
        self.toxicity_type_list = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
        if part == 1.:
            self.weight_dict = {"severe_toxicity": 1000, "obscene": 235, "identity_attack": 236, "insult": 22,
                            "threat": 646, "male": 45, "female": 35, "homosexual_gay_or_lesbian": 176, "christian": 50,
                            "jewish": 249, "muslim": 91, "black": 130, "white": 75, "psychiatric_or_mental_illness": 442,
                            "pp": 101, "np": 13, "pn": 20, "nn": 1,
                            "pp_male": 431, "np_male": 50, "pn_male": 17, "nn_male": 1,
                            "pp_female": 384, "np_female": 39, "pn_female": 17, "nn_female": 1,
                            "pp_homosexual_gay_or_lesbian": 900, "np_homosexual_gay_or_lesbian": 219, "pn_homosexual_gay_or_lesbian": 17, "nn_homosexual_gay_or_lesbian": 1,
                            "pp_christian": 859, "np_christian": 54, "pn_christian": 17, "nn_christian": 1,
                            "pp_jewish": 2365, "np_jewish": 278, "pn_jewish": 17, "nn_jewish": 1,
                            "pp_muslim": 606, "np_muslim": 108, "pn_muslim": 17, "nn_muslim": 1,
                            "pp_black": 586, "np_black": 167, "pn_black": 17, "nn_black": 1,
                            "pp_white": 387, "np_white": 94, "pn_white": 17, "nn_white": 1,
                            "pp_psychiatric_or_mental_illness": 2874, "np_psychiatric_or_mental_illness": 523, "pn_psychiatric_or_mental_illness": 17, "nn_psychiatric_or_mental_illness": 1}
        else:
            self.weight_dict = {"severe_toxicity": 1000, "obscene": 196, "identity_attack": 278, "insult": 22,
                            "threat": 609, "male": 45, "female": 33, "homosexual_gay_or_lesbian": 198, "christian": 48,
                            "jewish": 243, "muslim": 133, "black": 131, "white": 90, "psychiatric_or_mental_illness": 369,
                            "pp": 107, "np": 13, "pn": 19, "nn": 1,
                            "pp_male": 434, "np_male": 51, "pn_male": 17, "nn_male": 1,
                            "pp_female": 324, "np_female": 37, "pn_female": 17, "nn_female": 1,
                            "pp_homosexual_gay_or_lesbian": 1055, "np_homosexual_gay_or_lesbian": 244, "pn_homosexual_gay_or_lesbian": 17, "nn_homosexual_gay_or_lesbian": 1,
                            "pp_christian": 986, "np_christian": 50, "pn_christian": 17, "nn_christian": 1,
                            "pp_jewish": 2680, "np_jewish": 268, "pn_jewish": 16, "nn_jewish": 1,
                            "pp_muslim": 772, "np_muslim": 161, "pn_muslim": 17, "nn_muslim": 1,
                            "pp_black": 633, "np_black": 165, "pn_black": 17, "nn_black": 1,
                            "pp_white": 465, "np_white": 111, "pn_white": 17, "nn_white": 1,
                            "pp_psychiatric_or_mental_illness": 2748, "np_psychiatric_or_mental_illness": 427, "pn_psychiatric_or_mental_illness": 16, "nn_psychiatric_or_mental_illness": 1}
        self.stopwords = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
        self.seed_everything()
        self.max_len = 220
        self.epochs = epochs
        self.batch_size = batch_size
        self.split_ratio = 0.95
        self.sample_num = 1804874
        if not self.debug_mode:
            self.train_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/predict.csv").sample(int(self.sample_num * part), random_state=1234).fillna(0.)
            self.test_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")
        else:
            self.train_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/predict.csv").head(1000).fillna(0.)
            self.test_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv").head(1000)
        self.train_len = int(len(self.train_df) * self.split_ratio)
        self.evaluator = self.init_evaluator()

    def seed_everything(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
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
            target_weight += (~self.train_df["target"]) * np.where(self.train_df[self.identity_list].sum(axis=1) > 0, 1, 0) * 10
            target_weight += self.train_df["target"] * np.where(self.train_df[self.identity_list].sum(axis=1) == 0, 1, 0) * 10
        target_weight /= target_weight.mean()
        # 只留训练集
        target_weight = np.array(target_weight)
        target_weight = target_weight[:self.train_len]
        aux_weight = aux_weight[:self.train_len, :]
        identity_weight = identity_weight[:self.train_len, :]
        return target_weight, aux_weight, identity_weight

    def create_emb_weights(self, word_index):
        # 构建词向量字典
        with open("../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec", "r") as f:
            fasttext_emb_dict = {}
            for i, line in enumerate(f):
                if i == 1000 and self.debug_mode: break
                split = line.strip().split(" ")
                word = split[0]
                if word not in word_index: continue
                emb = np.array([float(num) for num in split[1:]])
                fasttext_emb_dict[word] = emb
        with open("../input/glove840b300dtxt/glove.840B.300d.txt", "r") as f:
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

    def custom_loss(self, y_pred, y_batch, epoch, target_weight=1., aux_weight=1., identity_weight=1.):
        target_pred = y_pred[:, 0]
        target_true = y_batch[:, 0]
        aux_pred = y_pred[:, 1: 6]
        aux_true = y_batch[:, 1: 6]
        identity_pred = y_pred[:, 6:]
        identity_true = y_batch[:, 6:]
        if epoch > 9:
            target_loss = FocalLoss()(target_pred, target_true)
        else:
            target_loss = nn.BCEWithLogitsLoss(reduction="none")(target_pred, target_true)
        target_loss = torch.mean(target_loss * target_weight)
        if epoch > 9:
            aux_loss = FocalLoss()(aux_pred, aux_true)
        else:
            aux_loss = nn.BCEWithLogitsLoss(reduction="none")(aux_pred, aux_true)
        aux_loss = torch.mean(aux_loss * aux_weight)
        if epoch > 9:
            identity_loss = FocalLoss()(identity_pred, identity_true)
        else:
            identity_loss = nn.BCEWithLogitsLoss(reduction="none")(identity_pred, identity_true)
        identity_loss = torch.mean(identity_loss * identity_weight)
        return target_loss, aux_loss, identity_loss

    def train(self):
        if self.debug_mode: self.epochs = 1
        # 加载 dataloader
        train_loader, valid_loader, tokenizer = self.create_dataloader()
        # 生成 embedding
        word_embedding = self.create_emb_weights(tokenizer.word_index)
        # 训练
        self.seed_everything()
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
            if epoch <= 10:
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
                target_loss, aux_loss, identity_loss = self.custom_loss(y_pred, y_batch, epoch, target_weight_batch, aux_weight_batch, identity_weight_batch)
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
            print("epoch: %d duration: %d min auc_score: %.4f" % (epoch, int((time.time() - start_time) / 60), auc_score))
            if not self.debug_mode and epoch > 0:
                temp_dict = model.state_dict()
                del temp_dict['embedding.weight']
                torch.save(temp_dict, "model[pytorch][%d][%s][%d][%.4f].bin" % (self.seed, self.model_name, epoch, auc_score))
        # del 训练相关输入和模型
        training_history = [train_loader, valid_loader, tokenizer, word_embedding, model, optimizer, scheduler]
        for variable in training_history:
            del variable
        gc.collect()


print("train16_correct_weight.py")
trainer = Trainer(model_name="train10_focal_loss_seed_kernel", epochs=25, batch_size=512, part=1., seed=1234, debug_mode=False)
trainer.train()

"""
fasttext-crawl-300d-2m
glove840b300dtxt
"""