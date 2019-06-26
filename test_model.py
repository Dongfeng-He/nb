import os
import pandas as pd
import random
import torch
from torch import nn
from torch.utils import data
import numpy as np
from torch.nn import functional as F
import time
from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_pretrained_bert import BertConfig
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from sklearn.metrics import roc_auc_score
import shutil
import copy
import math


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


class BertNeuralNet_simple(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNeuralNet_simple, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear_out = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        bert_output = self.dropout(pooled_output)
        out = self.linear_out(bert_output)
        return out


class BertNeuralNet_aux(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNeuralNet_aux, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        dense_size = config.hidden_size
        # 输出层
        self.linear_out = nn.Linear(dense_size, 1)
        self.linear_aux_out = nn.Linear(dense_size, 5)
        self.linear_identity_out = nn.Linear(dense_size, 9)
        self.linear_np_out = nn.Linear(dense_size, 4)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        bert_output = self.dropout(pooled_output)
        # 拼接
        hidden = bert_output
        # 输出层，用 sigmoid 就用 BCELoss，不用 sigmoid 就用 BCEWithLogitsLoss
        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        identity_result = self.linear_identity_out(hidden)
        np_result = self.linear_np_out(hidden)
        out = torch.cat([result, aux_result, identity_result, np_result], 1)
        return out


class BertNeuralNet_gate(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNeuralNet_gate, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        dense_size = config.hidden_size
        # 全连接层
        self.linear1 = nn.Linear(config.hidden_size, dense_size)
        self.linear2 = nn.Linear(config.hidden_size, dense_size)
        self.linear_gate = nn.Linear(config.hidden_size + dense_size, config.hidden_size)
        # 输出层
        self.linear_out = nn.Linear(dense_size, 1)
        self.linear_aux_out = nn.Linear(dense_size, 5)
        self.linear_identity_out = nn.Linear(dense_size, 9)
        self.linear_np_out = nn.Linear(dense_size, 4)
        self.linear_identity_hidden = nn.Linear(config.hidden_size, dense_size)
        self.bn1 = nn.BatchNorm1d(dense_size)
        self.bn2 = nn.BatchNorm1d(dense_size)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        bert_output = self.dropout(pooled_output)

        # 全连接层
        identity_hidden = self.linear_identity_hidden(bert_output)
        identity_hidden = F.relu(identity_hidden)
        #identity_hidden = self.bn1(identity_hidden)
        identity_hidden = F.dropout(identity_hidden, p=0.3)
        identity_result = self.linear_identity_out(identity_hidden)
        identity_hidden_conc = torch.cat((bert_output, identity_hidden), 1)
        gate_hidden = self.linear_gate(identity_hidden_conc)
        #gate_hidden = self.bn2(gate_hidden)
        gate = torch.sigmoid(gate_hidden)
        #gate = F.dropout(gate, p=0.3)
        bert_output_gate = bert_output * gate

        # 拼接
        hidden = bert_output_gate
        # 输出层，用 sigmoid 就用 BCELoss，不用 sigmoid 就用 BCEWithLogitsLoss
        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        np_result = self.linear_np_out(hidden)
        out = torch.cat([result, aux_result, identity_result, np_result], 1)
        return out


class Predictor:
    def __init__(self, model_name, epochs=1, batch_size=64, base_batch_size=32, part=1., half=2, seed=1234, debug_mode=False):
        self.device = torch.device('cuda')
        self.input_dir = "../input"
        self.work_dir = "../working/"
        self.debug_mode = debug_mode
        self.model_name = model_name
        self.half = half
        self.seed = seed
        self.identity_list = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
        self.toxicity_type_list = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
        self.stopwords = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
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
        self.seed_everything()
        self.max_len = 220
        self.epochs = epochs
        self.base_batch_size = base_batch_size
        self.batch_size = batch_size
        self.split_ratio = 0.95
        self.sample_num = 1804874
        if not self.debug_mode:
            self.train_df = pd.read_csv(os.path.join("/root/nb/data/train.csv")).sample(int(self.sample_num * part), random_state=1234).fillna(0.)
            self.test_df = pd.read_csv(os.path.join("/root/nb/data/test.csv"))
        else:
            self.train_df = pd.read_csv(os.path.join("/root/nb/data/train.csv")).head(1000).fillna(0.)
            self.test_df = pd.read_csv(os.path.join("/root/nb/data/test.csv")).head(1000)
        self.train_len = int(len(self.train_df) * self.split_ratio)
        self.test_len = len(self.test_df)
        self.evaluator = self.init_evaluator()
        self.bert_config = BertConfig('/root/nb/data/uncased_L-12_H-768_A-12/'+'bert_config.json')
        self.bert_model_path = '/root/nb/data/uncased_L-12_H-768_A-12/'

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

    def convert_lines(self, text_series, max_seq_length, bert_tokenizer):
        max_seq_length -= 2
        all_tokens = []
        for text in text_series:
            tokens = bert_tokenizer.tokenize(text)
            if len(tokens) > max_seq_length:
                tokens = tokens[:max_seq_length]
            one_token = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"]) + [0] * (max_seq_length - len(tokens))
            all_tokens.append(one_token)
        return np.array(all_tokens)

    def create_dataloader(self):
        # 读取输入输出
        print("Load data")
        train_comments = self.train_df["comment_text"].astype(str)
        train_label = self.train_df["target"].values
        train_type_labels = self.train_df[self.toxicity_type_list].values

        # 新的 np 任务
        train_np_labels = np.zeros((len(self.train_df), 4))
        train_np_identity_labels = np.zeros((len(self.train_df), len(self.identity_list) * 4))
        train_df_copy = self.train_df[self.identity_list + ["target"]]
        for column in self.identity_list + ["target"]:
            train_df_copy[column] = np.where(train_df_copy[column] > 0.5, True, False)
        pp_label_bool = train_df_copy["target"] & np.where(train_df_copy[self.identity_list].sum(axis=1) > 0, True, False)
        np_label_bool = ~train_df_copy["target"] & np.where(train_df_copy[self.identity_list].sum(axis=1) > 0, True, False)
        pn_label_bool = train_df_copy["target"] & np.where((train_df_copy[self.identity_list]).sum(axis=1) == 0, True, False)
        nn_label_bool = ~train_df_copy["target"] & np.where((train_df_copy[self.identity_list]).sum(axis=1) == 0, True, False)
        train_np_labels[:, 0] = np.where(pp_label_bool > 0, 1, 0)
        train_np_labels[:, 1] = np.where(np_label_bool > 0, 1, 0)
        train_np_labels[:, 2] = np.where(pn_label_bool > 0, 1, 0)
        train_np_labels[:, 3] = np.where(nn_label_bool > 0, 1, 0)
        for i, column in enumerate(self.identity_list):
            pp_label_bool = train_df_copy["target"] & train_df_copy[column]
            np_label_bool = ~train_df_copy["target"] & train_df_copy[column]
            pn_label_bool = train_df_copy["target"] & (~train_df_copy[column])
            nn_label_bool = ~train_df_copy["target"] & (~train_df_copy[column])
            train_np_identity_labels[:, i * 4 + 0] = np.where(pp_label_bool > 0, 1, 0)
            train_np_identity_labels[:, i * 4 + 1] = np.where(np_label_bool > 0, 1, 0)
            train_np_identity_labels[:, i * 4 + 2] = np.where(pn_label_bool > 0, 1, 0)
            train_np_identity_labels[:, i * 4 + 3] = np.where(nn_label_bool > 0, 1, 0)

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
        print("Init tokenizer")
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_path, cache_dir=None, do_lower_case=True)
        print("Tokenizing")
        train_bert_tokens = self.convert_lines(self.train_df["comment_text"].fillna("DUMMY_VALUE"), self.max_len, bert_tokenizer)
        # 划分训练集和验证集
        valid_tokens = train_bert_tokens[self.train_len:]
        valid_label = train_label[self.train_len:]
        valid_type_labels = train_type_labels[self.train_len:]
        train_tokens = train_bert_tokens[: self.train_len]
        train_label = train_label[: self.train_len]
        train_type_labels = train_type_labels[: self.train_len]
        valid_identity_type_labels = train_identity_type_labels[self.train_len:]
        train_identity_type_labels = train_identity_type_labels[: self.train_len]
        valid_identity_type_binary_lables = train_identity_type_binary_lables[self.train_len:]
        train_identity_type_binary_lables = train_identity_type_binary_lables[: self.train_len]
        valid_identity_sum_label = train_identity_sum_label[self.train_len:]
        train_identity_sum_label = train_identity_sum_label[: self.train_len]
        valid_identity_binary_label = train_identity_binary_label[self.train_len:]
        train_identity_binary_label = train_identity_binary_label[: self.train_len]
        valid_np_labels = train_np_labels[self.train_len:]
        train_np_labels = train_np_labels[: self.train_len]
        valid_np_identity_labels = train_np_identity_labels[self.train_len:]
        train_np_identity_labels = train_np_identity_labels[: self.train_len]

        # 计算样本权重
        target_weight, aux_weight, identity_weight, np_weight, np_identity_weight = self.cal_sample_weights()

        # 将符号化数据转成 tensor
        train_x_tensor = torch.tensor(train_tokens, dtype=torch.long)
        valid_x_tensor = torch.tensor(valid_tokens, dtype=torch.long)
        train_y_tensor = torch.tensor(np.hstack([train_label[:, np.newaxis], train_type_labels, train_identity_type_labels, train_np_labels]), dtype=torch.float32)
        valid_y_tensor = torch.tensor(np.hstack([valid_label[:, np.newaxis], valid_type_labels, valid_identity_type_labels, valid_np_labels]), dtype=torch.float32)
        target_weight_tensor = torch.tensor(target_weight, dtype=torch.float32)
        aux_weight_tensor = torch.tensor(aux_weight, dtype=torch.float32)
        identity_weight_tensor = torch.tensor(identity_weight, dtype=torch.float32)
        np_weight_tensor = torch.tensor(np_weight, dtype=torch.float32)
        np_identity_weight_tensor = torch.tensor(np_identity_weight, dtype=torch.float32)
        if torch.cuda.is_available():
            train_x_tensor = train_x_tensor.to(self.device)
            valid_x_tensor = valid_x_tensor.to(self.device)
            train_y_tensor = train_y_tensor.to(self.device)
            valid_y_tensor = valid_y_tensor.to(self.device)
            target_weight_tensor = target_weight_tensor.to(self.device)
            aux_weight_tensor = aux_weight_tensor.to(self.device)
            identity_weight_tensor = identity_weight_tensor.to(self.device)
            np_weight_tensor = np_weight_tensor.cuda()
            np_identity_weight_tensor = np_identity_weight_tensor.cuda()
        # 将 tensor 转成 dataset，训练数据和标签一一对应，用 dataloader 加载的时候 dataset[:-1] 是 x，dataset[-1] 是 y
        train_dataset = data.TensorDataset(train_x_tensor, train_y_tensor, target_weight_tensor, aux_weight_tensor, identity_weight_tensor, np_weight_tensor, np_identity_weight_tensor)
        valid_dataset = data.TensorDataset(valid_x_tensor, valid_y_tensor)
        # 将 dataset 转成 dataloader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.base_batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.base_batch_size, shuffle=False)
        # 返回训练数据
        return train_loader, valid_loader

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
        # np weight
        np_weight = np.zeros((len(self.train_df), 4))
        np_identity_weight = np.zeros((len(self.train_df), len(self.identity_list) * 4))
        train_df_copy = self.train_df[self.identity_list + ["target"]]
        for column in self.identity_list + ["target"]:
            train_df_copy[column] = np.where(train_df_copy[column] > 0.5, True, False)
        pp_label_bool = train_df_copy["target"] & np.where(train_df_copy[self.identity_list].sum(axis=1) > 0, True, False)
        np_label_bool = ~train_df_copy["target"] & np.where(train_df_copy[self.identity_list].sum(axis=1) > 0, True, False)
        pn_label_bool = train_df_copy["target"] & np.where((train_df_copy[self.identity_list]).sum(axis=1) == 0, True, False)
        nn_label_bool = ~train_df_copy["target"] & np.where((train_df_copy[self.identity_list]).sum(axis=1) == 0, True, False)
        np_weight[:, 0] = np.where(pp_label_bool > 0, 1, 1)
        np_weight[:, 1] = np.where(np_label_bool > 0, 1, 1)
        np_weight[:, 2] = np.where(pn_label_bool > 0, 1, 1)
        np_weight[:, 3] = np.where(nn_label_bool > 0, 1, 1)
        for i, column in enumerate(self.identity_list):
            pp_label_bool = train_df_copy["target"] & train_df_copy[column]
            np_label_bool = ~train_df_copy["target"] & train_df_copy[column]
            pn_label_bool = train_df_copy["target"] & (~train_df_copy[column])
            nn_label_bool = ~train_df_copy["target"] & (~train_df_copy[column])
            np_identity_weight[:, i * 4 + 0] = np.where(pp_label_bool > 0, self.weight_dict["pp_%s" % column], 1)
            np_identity_weight[:, i * 4 + 1] = np.where(np_label_bool > 0, self.weight_dict["np_%s" % column], 1)
            np_identity_weight[:, i * 4 + 2] = np.where(pn_label_bool > 0, self.weight_dict["pn_%s" % column], 1)
            np_identity_weight[:, i * 4 + 3] = np.where(nn_label_bool > 0, self.weight_dict["nn_%s" % column], 1)
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
        target_weight = np.array(target_weight)
        target_weight = target_weight[: self.train_len]
        aux_weight = aux_weight[: self.train_len, :]
        identity_weight = identity_weight[: self.train_len, :]
        np_weight = np_weight[: self.train_len, :]
        np_identity_weight = np_identity_weight[: self.train_len, :]
        return target_weight, aux_weight, identity_weight, np_weight, np_identity_weight

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self):
        if self.debug_mode: self.epochs = 1
        # 加载 dataloader
        train_loader, valid_loader = self.create_dataloader()
        # 训练
        self.seed_everything()
        # 加载预训练模型
        model_name_list = ["model_train2_simple_target_1234_2_18_0.9417.bin",
                           "model_train11_np_task_1234_2_5_40min_5min_0.9421.bin",
                           "model_train12_aux_new_1234_2_5_38min_5min_0.9426.bin",
                           "model_train13_aux_mean_42_2_5_38min_5min_0.9428.bin",
                           "model_train14_all_77_2_4_138min_5min_0.9423.bin"]
        print("Load checkpoint")
        # TODO: 读取模型
        y_pred_list = []
        for model_name in model_name_list:
            valid_start_time = time.time()
            if model_name in ["model_train2_simple_target_1234_2_18_0.9417.bin"]:
                model = BertNeuralNet_simple(self.bert_config)
            elif model_name in ["model_train11_np_task_1234_2_5_40min_5min_0.9421.bin"]:
                model = BertNeuralNet_gate(self.bert_config)
            elif model_name in ["model_train12_aux_1234_2_5_12min_3min_0.9418.bin",
                           "model_train12_aux_new_1234_2_5_38min_5min_0.9426.bin",
                           "model_train13_aux_mean_42_2_5_38min_5min_0.9428.bin"]:
                model = BertNeuralNet_aux(self.bert_config)
            elif model_name in ["model_train14_all_77_2_4_138min_5min_0.9423.bin"]:
                model = BertNeuralNet_aux(self.bert_config)
            model.load_state_dict(torch.load("/root/nb/data/model2/" + model_name))
            model = model.to(self.device)
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            y_pred = np.zeros((len(self.train_df) - self.train_len))
            copy_start_time = time.time()
            new_valid_loader = copy.deepcopy(valid_loader)
            copy_duration = int((time.time() - copy_start_time) / 60)
            for j, test_batch_data in enumerate(new_valid_loader):
                x_batch = test_batch_data[0]
                batch_y_pred = self.sigmoid(model(x_batch.to(self.device), attention_mask=(x_batch > 0).to(self.device), labels=None).detach().cpu().numpy())[:, 0]
                y_pred[j * self.base_batch_size: (j + 1) * self.base_batch_size] = batch_y_pred
            y_pred_list.append(y_pred)
            valid_duration = int((time.time() - valid_start_time) / 60)
            print("valid_duration", valid_duration, "copy_duration", copy_duration)
        # 每个单模型的分数
        for j in range(len(model_name_list)):
            auc_score = self.evaluator.get_final_metric(y_pred_list[j])
            print(model_name_list[j], auc_score)
        # 混合模型分数
        best_auc_score = 0
        random.seed(1234)
        for j in range(5000):
            num_list = np.array([random.uniform(0, 1) for _ in range(len(model_name_list))])
            num_list /= sum(num_list)
            y_pred_mean = np.zeros(y_pred_list[0].shape)
            for i in range(len(model_name_list)):
                y_pred_mean += y_pred_list[i] * num_list[i]
            auc_score = self.evaluator.get_final_metric(y_pred_mean)
            if auc_score > best_auc_score:
                print("auc_score:", auc_score, "num_list:", num_list)
                best_auc_score = auc_score


if __name__ == "__main__":
    print("train50_bnew_simple_target_inference_blend_test_2.py")
    predictor = Predictor("bert", epochs=1, batch_size=64, base_batch_size=32, part=1., half=2, seed=1234, debug_mode=False)
    predictor.predict()
