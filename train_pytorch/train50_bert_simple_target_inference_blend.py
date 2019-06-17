import os
import pandas as pd
import random
import torch
from torch import nn
from torch.utils import data
import numpy as np
import time
from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_pretrained_bert import BertConfig
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from sklearn.metrics import roc_auc_score
import shutil
import copy


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


class BertNeuralNet(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNeuralNet, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear_out = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        bert_output = self.dropout(pooled_output)
        out = self.linear_out(bert_output)
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
        self.seed_everything()
        self.max_len = 220
        self.epochs = epochs
        self.base_batch_size = base_batch_size
        self.batch_size = batch_size
        self.split_ratio = 0.95
        self.sample_num = 1804874
        if not self.debug_mode:
            self.train_df = pd.read_csv(os.path.join("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")).sample(int(self.sample_num * part), random_state=1234).fillna(0.)
            self.test_df = pd.read_csv(os.path.join("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv"))
        else:
            self.train_df = pd.read_csv(os.path.join("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")).head(1000).fillna(0.)
            self.test_df = pd.read_csv(os.path.join("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")).head(1000)
        self.train_len = int(len(self.train_df) * self.split_ratio)
        self.test_len = len(self.test_df)
        self.bert_config = BertConfig('../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'+'bert_config.json')
        self.bert_model_path = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'

    def seed_everything(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

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
        test_comments = self.test_df["comment_text"].astype(str)
        # tokenizer 训练
        print("Init tokenizer")
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_path, cache_dir=None, do_lower_case=True)
        print("Tokenizing")
        test_bert_tokens = self.convert_lines(self.test_df["comment_text"].fillna("DUMMY_VALUE"), self.max_len, bert_tokenizer)
        # 将符号化数据转成 tensor
        test_x_tensor = torch.tensor(test_bert_tokens, dtype=torch.long)
        if torch.cuda.is_available():
            test_x_tensor = test_x_tensor.to(self.device)
        test_dataset = data.TensorDataset(test_x_tensor)
        # 将 dataset 转成 dataloader
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.base_batch_size, shuffle=False)
        # 返回训练数据
        return test_loader

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self):
        if self.debug_mode: self.epochs = 1
        # 加载 dataloader
        test_loader = self.create_dataloader()
        # 训练
        self.seed_everything()
        """
        # 预训练 bert 转成 pytorch
        if os.path.exists(self.work_dir + 'pytorch_model.bin') is False:
            print("Convert pre-trained model")
            convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
                self.bert_model_path + 'bert_model.ckpt',
                self.bert_model_path + 'bert_config.json',
                self.work_dir + 'pytorch_model.bin')
        shutil.copyfile(self.bert_model_path + 'bert_config.json', self.work_dir + 'bert_config.json')
        """
        # 加载预训练模型
        model_name_list = ["model_bert_1234_2_12_train2_simple_target_0.9410.bin",
                           "model_bert_1234_2_14_train2_simple_target_0.9417.bin",
                           "model_bert_1234_2_15_train2_simple_target_0.9410.bin",
                           "model_bert_1234_2_17_train2_simple_target_0.9419.bin",
                           "model_bert_1234_2_18_train2_simple_target_0.9417.bin"]
        print("Load checkpoint")
        model = BertNeuralNet(self.bert_config)
        # TODO: 读取模型
        y_pred_list = []
        for model_name in model_name_list:
            valid_start_time = time.time()
            model.load_state_dict(torch.load("../input/bert-model/bert-model/bert-model/" + model_name))
            model = model.to(self.device)
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            y_pred = np.zeros((self.test_len))
            copy_start_time = time.time()
            new_test_loader = copy.deepcopy(test_loader)
            copy_duration = int((time.time() - copy_start_time) / 60)
            for j, test_batch_data in enumerate(new_test_loader):
                x_batch = test_batch_data[0]
                batch_y_pred = self.sigmoid(model(x_batch.to(self.device), attention_mask=(x_batch > 0).to(self.device), labels=None).detach().cpu().numpy())[:, 0]
                y_pred[j * self.base_batch_size: (j + 1) * self.base_batch_size] = batch_y_pred
            y_pred_list.append(y_pred)
            valid_duration = int((time.time() - valid_start_time) / 60)
            print("valid_duration", valid_duration, "copy_duration", copy_duration)
        y_pred_mean = np.mean(y_pred_list, axis=0)
        submission = pd.DataFrame.from_dict({
            'id': self.test_df['id'],
            'prediction': y_pred_mean
        })
        submission.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    print("train50_bert_simple_target_inference_blend.py")
    predictor = Predictor("bert", epochs=1, batch_size=64, base_batch_size=32, part=1., half=2, seed=1234, debug_mode=False)
    predictor.predict()
