import os
import pandas as pd
from evaluation import *
import random
import copy
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
import numpy as np
import time
import math
import gc
from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from pytorch_pretrained_bert import BertTokenizer, BertAdam, BertModel
from pytorch_pretrained_bert import BertConfig
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from apex import amp


class BertNeuralNet(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNeuralNet, self).__init__(config)
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

        result = self.linear_out(bert_output)
        identity_result = self.linear_identity_out(bert_output)
        aux_result = self.linear_aux_out(bert_output)
        np_result = self.linear_np_out(bert_output)
        out = torch.cat([result, aux_result, identity_result, np_result], 1)
        return out


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


class Trainer:
    def __init__(self, data_dir, model_name, epochs=4, batch_size=64, base_batch_size=32, part=1., seed=1234, debug_mode=False):
        self.device = torch.device('cuda')
        self.data_dir = data_dir
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
        self.base_batch_size = base_batch_size
        self.batch_size = batch_size
        self.split_ratio = 0.80
        self.sample_num = 1804874
        if not self.debug_mode:
            self.train_df = pd.read_csv(os.path.join(self.data_dir, "train.csv")).sample(int(self.sample_num * part), random_state=1234).fillna(0.)
            self.test_df = pd.read_csv(os.path.join(self.data_dir, "test.csv"))
        else:
            self.train_df = pd.read_csv(os.path.join(self.data_dir, "train.csv")).head(1000).fillna(0.)
            self.test_df = pd.read_csv(os.path.join(self.data_dir, "test.csv")).head(1000)
        self.train_len = int(len(self.train_df) * self.split_ratio)
        self.evaluator = self.init_evaluator()
        self.bert_config = BertConfig(os.path.join(self.data_dir, "uncased_L-12_H-768_A-12/bert_config.json"))
        self.bert_model_path = os.path.join(self.data_dir, "uncased_L-12_H-768_A-12/")

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
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_path, cache_dir=None, do_lower_case=True)
        train_bert_tokens = self.convert_lines(self.train_df["comment_text"].fillna("DUMMY_VALUE"), self.max_len, bert_tokenizer)
        # 划分训练集和验证集
        valid_tokens = train_bert_tokens[self.train_len:]
        valid_label = train_label[self.train_len:]
        valid_type_labels = train_type_labels[self.train_len:]
        train_tokens = train_bert_tokens[:self.train_len]
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
        valid_np_labels = train_np_labels[self.train_len:]
        train_np_labels = train_np_labels[:self.train_len]
        valid_np_identity_labels = train_np_identity_labels[self.train_len:]
        train_np_identity_labels = train_np_identity_labels[:self.train_len]


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
        train_attention_mask_tensor = train_x_tensor > 0
        valid_attention_mask_tensor = valid_x_tensor > 0
        if torch.cuda.is_available():
            train_x_tensor = train_x_tensor.to(self.device)
            valid_x_tensor = valid_x_tensor.to(self.device)
            train_y_tensor = train_y_tensor.to(self.device)
            valid_y_tensor = valid_y_tensor.to(self.device)
            target_weight_tensor = target_weight_tensor.to(self.device)
            aux_weight_tensor = aux_weight_tensor.to(self.device)
            identity_weight_tensor = identity_weight_tensor.to(self.device)
            train_attention_mask_tensor = train_attention_mask_tensor.to(self.device)
            valid_attention_mask_tensor = valid_attention_mask_tensor.to(self.device)
            np_weight_tensor = np_weight_tensor.to(self.device)
            np_identity_weight_tensor = np_identity_weight_tensor.to(self.device)
        # 将 tensor 转成 dataset，训练数据和标签一一对应，用 dataloader 加载的时候 dataset[:-1] 是 x，dataset[-1] 是 y
        train_dataset = data.TensorDataset(train_x_tensor, train_y_tensor, target_weight_tensor, aux_weight_tensor, identity_weight_tensor, train_attention_mask_tensor, np_weight_tensor)
        valid_dataset = data.TensorDataset(valid_x_tensor, valid_y_tensor, valid_attention_mask_tensor)
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
        target_weight = target_weight[:self.train_len]
        aux_weight = aux_weight[:self.train_len, :]
        identity_weight = identity_weight[:self.train_len, :]
        np_weight = np_weight[:self.train_len, :]
        np_identity_weight = np_identity_weight[:self.train_len, :]
        return target_weight, aux_weight, identity_weight, np_weight, np_identity_weight

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def custom_loss(self, y_pred, y_batch, epoch, target_weight=1., aux_weight=1., identity_weight=1., np_weight=1.):
        target_pred = y_pred[:, 0]
        target_true = y_batch[:, 0]
        aux_pred = y_pred[:, 1: 6]
        aux_true = y_batch[:, 1: 6]
        identity_pred = y_pred[:, 6: 15]
        identity_true = y_batch[:, 6: 15]
        np_pred = y_pred[:, 15: 19]
        np_true = y_batch[:, 15: 19]
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
        if epoch > 9:
            np_loss = FocalLoss()(np_pred, np_true)
        else:
            np_loss = nn.BCEWithLogitsLoss(reduction="none")(np_pred, np_true)
        np_loss = torch.mean(np_loss * np_weight)
        return target_loss, aux_loss, identity_loss, np_loss

    def train(self):
        if self.debug_mode: self.epochs = 1
        # 加载 dataloader
        train_loader, valid_loader = self.create_dataloader()
        # 训练
        self.seed_everything()
        lr = 2e-5
        accumulation_steps = math.ceil(self.batch_size / self.base_batch_size)
        # 预训练 bert 转成 pytorch
        if os.path.exists(self.bert_model_path + "pytorch_model.bin") is False:
            convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
                self.bert_model_path + 'bert_model.ckpt',
                self.bert_model_path + 'bert_config.json',
                self.bert_model_path + 'pytorch_model.bin')
        # 加载预训练模型
        model = BertNeuralNet.from_pretrained(self.bert_model_path, cache_dir=None)
        model.zero_grad()
        model = model.to(self.device)
        # 不同的参数组设置不同的 weight_decay
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        epoch_steps = int(self.train_len / self.base_batch_size / accumulation_steps)
        num_train_optimization_steps = int(self.epochs * epoch_steps)
        valid_every = math.floor(epoch_steps * accumulation_steps / 5)
        optimizer = BertAdam(optimizer_grouped_parameters, lr=lr, warmup=0.05, t_total=num_train_optimization_steps)
        # 渐变学习速率
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
        # 开始训练
        for epoch in range(self.epochs):
            train_start_time = time.time()
            model.train()
            optimizer.zero_grad()
            # 加载每个 batch 并训练
            for i, batch_data in enumerate(train_loader):
                x_batch = batch_data[0]
                y_batch = batch_data[1]
                target_weight_batch = batch_data[2]
                aux_weight_batch = batch_data[3]
                identity_weight_batch = batch_data[4]
                x_mask = batch_data[5]
                np_weight_batch = batch_data[6]
                y_pred = model(x_batch, attention_mask=x_mask, labels=None)
                target_loss, aux_loss, identity_loss, np_loss = self.custom_loss(y_pred, y_batch, epoch, target_weight_batch, aux_weight_batch, identity_weight_batch, np_weight_batch)
                loss = target_loss * 0.5 + identity_loss * 0.5
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                # 验证
                if (i + 1) % valid_every == 0:
                    valid_start_time = time.time()
                    model.eval()
                    y_pred = np.zeros((len(self.train_df) - self.train_len))
                    for j, valid_batch_data in enumerate(valid_loader):
                        x_batch = valid_batch_data[0]
                        x_mask = valid_batch_data[2]
                        batch_y_pred = self.sigmoid(model(x_batch, attention_mask=x_mask, labels=None).detach().cpu().numpy())[:, 0]
                        y_pred[j * self.base_batch_size: (j + 1) * self.base_batch_size] = batch_y_pred
                    # 计算得分
                    auc_score = self.evaluator.get_final_metric(y_pred)
                    print("epoch: %d duration: %d min auc_score: %.4f" % (epoch, int((time.time() - train_start_time) / 60), auc_score))
                    if not self.debug_mode:
                        state_dict = model.state_dict()
                        stage = int((i + 1) / valid_every)
                        train_duration = int((time.time() - train_start_time) / 60)
                        valid_duration = int((time.time() - valid_start_time) / 60)
                        train_start_time = time.time()
                        # model[bert][seed][epoch][stage][model_name][stage_train_duration][valid_duration][score].bin
                        model_name = "model2/model_%d_%d_%d_%s_%dmin_%dmin_%.4f.bin" % (self.seed, epoch + 1, stage, self.model_name, train_duration, valid_duration, auc_score)
                        torch.save(state_dict, os.path.join(self.data_dir, model_name))
                    model.train()
        # del 训练相关输入和模型
        training_history = [train_loader, valid_loader, model, optimizer, param_optimizer, optimizer_grouped_parameters]
        for variable in training_history:
            del variable
        gc.collect()


if __name__ == "__main__":
    data_dir = "/Users/hedongfeng/PycharmProjects/unintended_bias/data/"
    trainer = Trainer(data_dir, "model_name", debug_mode=True)
    trainer.train()
