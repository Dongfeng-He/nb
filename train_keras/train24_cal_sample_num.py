import os
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import *
from keras.preprocessing import text, sequence
from keras.callbacks import LearningRateScheduler
from evaluation import *
from keras import backend as K
import gc


class Trainer:
    def __init__(self, data_dir, model_name, debug_mode=False):
        self.data_dir = data_dir
        self.debug_mode = debug_mode
        self.model_name = model_name
        self.identity_list = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
        self.toxicity_type_list = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
        self.stopwords = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
        self.seed = 5
        self.max_len = 220
        self.split_ratio = 0.95
        if not self.debug_mode:
            self.train_df = pd.read_csv(os.path.join(self.data_dir, "train_keras.csv"))
            self.test_df = pd.read_csv(os.path.join(self.data_dir, "test.csv"))
        else:
            self.train_df = pd.read_csv(os.path.join(self.data_dir, "train_keras.csv")).head(1000)
            self.test_df = pd.read_csv(os.path.join(self.data_dir, "test.csv")).head(1000)
        self.train_len = int(len(self.train_df) * self.split_ratio)
        self.evaluator = self.init_evaluator()

    def init_evaluator(self):
        # 初始化评分函数类
        y_true = self.train_df['target'].values
        y_identity = self.train_df[self.identity_list].values
        valid_y_true = y_true[self.train_len:]
        valid_y_identity = y_identity[self.train_len:]
        evaluator = JigsawEvaluator(valid_y_true, valid_y_identity) # y_true 必须是0或1，不能是离散值
        return evaluator

    def create_train_data(self):
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
        train_identity_binary_label = train_identity_binary

        test_comments = self.test_df["comment_text"].astype(str)
        # tokenizer 训练
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

        # 划分身份标签
        valid_identity_type_labels = train_identity_type_labels[self.train_len:]
        train_identity_type_labels = train_identity_type_labels[:self.train_len]
        valid_identity_type_binary_lables = train_identity_type_binary_lables[self.train_len:]
        train_identity_type_binary_lables = train_identity_type_binary_lables[:self.train_len]
        valid_identity_sum_label = train_identity_sum_label[self.train_len:]
        train_identity_sum_label = train_identity_sum_label[:self.train_len]
        valid_identity_binary_label = train_identity_binary_label[self.train_len:]
        train_identity_binary_label = train_identity_binary_label[:self.train_len]

        # 数据集
        dataset = {"train_tokens": train_tokens,
                   "train_label": train_label,
                   "train_type_labels": train_type_labels,
                   "valid_tokens": valid_tokens,
                   "valid_label": valid_label,
                   "valid_type_labels": valid_type_labels,
                   "test_tokens": test_tokens,
                   "tokenizer": tokenizer,
                   "valid_identity_type_labels": valid_identity_type_labels,
                   "train_identity_type_labels": train_identity_type_labels,
                   "valid_identity_type_binary_lables": valid_identity_type_binary_lables,
                   "train_identity_type_binary_lables": train_identity_type_binary_lables,
                   "valid_identity_sum_label": valid_identity_sum_label,
                   "train_identity_sum_label": train_identity_sum_label,
                   "valid_identity_binary_label": valid_identity_binary_label,
                   "train_identity_binary_label": train_identity_binary_label}
        return dataset

    def cal_sample_weights(self, percent):
        train_df = copy.deepcopy(self.train_df.head(int(len(self.train_df) * percent)))
        sample_num_dict = {}
        for column in self.toxicity_type_list + self.identity_list + ["target"]:
            sample_num_dict[column] = np.sum(np.where(train_df[column] > 0.5, 1, 0))
        # 把 train_df 中的 target 和 所有身份的值变成 bool
        for column in self.identity_list + ["target"]:
            train_df[column] = np.where(train_df[column] > 0.5, True, False)
        sample_num_dict["np"] = np.sum((~train_df["target"]) * np.where(train_df[self.identity_list].sum(axis=1) > 0, 1, 0))
        sample_num_dict["pn"] = np.sum(train_df["target"] * np.where((~train_df[self.identity_list]).sum(axis=1) > 0, 1, 0))
        print("percent: %f total: %d" % (percent, len(train_df)))
        for key, value in sample_num_dict.items():
            print(key, value, int((len(train_df) - value) / value))
        print("")

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

    def build_model(self, emb_weights):
        def hidden_layer(layer_input, layer_size, init, activation, dropout_rate=0.5):
            output = Dense(layer_size, kernel_initializer=init)(layer_input)
            output = BatchNormalization()(output)
            output = Activation(activation)(output)
            output = Dropout(dropout_rate)(output)
            return output

        if not self.debug_mode:
            rnn_size = 256
            hidden_size = 512
        else:
            rnn_size = 2
            hidden_size = 2
        # 嵌入层
        token_input = Input(shape=(self.max_len,), dtype="int32")
        embedding_layer = Embedding(input_dim=emb_weights.shape[0],
                                    output_dim=emb_weights.shape[1],
                                    weights=[emb_weights],
                                    trainable=True)
        token_emb = embedding_layer(token_input)
        emb_model = Model(token_input, token_emb)

        # 网络层


        emb_input = Input(shape=(self.max_len, 600,), dtype="float32")


        token_emb = SpatialDropout1D(0.2)(token_emb)
        output1 = Bidirectional(CuDNNGRU(rnn_size, return_sequences=True))(token_emb)
        output2 = Bidirectional(CuDNNLSTM(rnn_size, return_sequences=True))(token_emb)
        output1 = GlobalMaxPooling1D()(output1)
        output2 = GlobalMaxPooling1D()(output2)
        # 拼接
        output = concatenate([output1, output2])
        # 全连接层
        output = hidden_layer(output, hidden_size, "he_normal", "relu")
        output = hidden_layer(output, hidden_size, "he_normal", "relu")
        # 输出层
        output1 = Dense(1, activation="sigmoid")(output)
        output2 = Dense(6, activation="sigmoid")(output)
        model = Model(token_input, [output1, output2])
        model.compile(optimizer="adam",
                      loss="binary_crossentropy",
                      metrics=["acc"])
        model.summary()
        return model

    def train(self, epochs=5, batch_size=16):
        if self.debug_mode: epochs = 1
        #dataset = self.create_dataloader()
        self.cal_sample_weights(0.2)
        self.cal_sample_weights(0.3)
        self.cal_sample_weights(0.5)
        self.cal_sample_weights(1)


if __name__ == "__main__":
    data_dir = "/Users/hedongfeng/PycharmProjects/unintended_bias/data/"
    trainer = Trainer(data_dir, "model_name", False)
    trainer.train(batch_size=16)

    """
    percent: 0.200000 total: 360974
    target 21180 16
    severe_toxicity 1 360973
    obscene 1945 184
    identity_attack 1190 302
    insult 15593 22
    threat 641 562
    male 8191 43
    female 11136 31
    homosexual_gay_or_lesbian 1849 194
    christian 7380 47
    jewish 991 363
    muslim 2035 176
    black 3015 118
    white 4386 81
    psychiatric_or_mental_illness 902 399
    np 26656 12
    pn 21180 16
    
    percent: 0.300000 total: 541462
    target 32202 15
    severe_toxicity 1 541461
    obscene 2753 195
    identity_attack 1946 277
    insult 24052 21
    threat 889 608
    male 11809 44
    female 16285 32
    homosexual_gay_or_lesbian 2728 197
    christian 11167 47
    jewish 2221 242
    muslim 4064 132
    black 4129 130
    white 6005 89
    psychiatric_or_mental_illness 1464 368
    np 40053 12
    pn 32202 15
    
    percent: 0.500000 total: 902437
    target 51701 16
    severe_toxicity 1 902436
    obscene 3984 225
    identity_attack 3498 256
    insult 38665 22
    threat 1348 668
    male 19398 45
    female 26472 33
    homosexual_gay_or_lesbian 4429 202
    christian 18894 46
    jewish 4248 211
    muslim 11600 76
    black 5780 155
    white 9323 95
    psychiatric_or_mental_illness 2151 418
    np 68599 12
    pn 51701 16
    
    percent: 1.000000 total: 1804874
    target 106438 15
    severe_toxicity 8 225608
    obscene 7648 234
    identity_attack 7633 235
    insult 79887 21
    threat 2793 645
    male 40036 44
    female 50548 34
    homosexual_gay_or_lesbian 10233 175
    christian 35507 49
    jewish 7239 248
    muslim 19666 90
    black 13869 129
    white 23852 74
    psychiatric_or_mental_illness 4077 441
    np 135896 12
    pn 106438 15
    """