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
            self.train_df = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
            self.test_df = pd.read_csv(os.path.join(self.data_dir, "test.csv"))
        else:
            self.train_df = pd.read_csv(os.path.join(self.data_dir, "train.csv")).head(1000)
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
        # 数据集
        dataset = {"train_tokens": train_tokens,
                   "train_label": train_label,
                   "train_type_labels": train_type_labels,
                   "valid_tokens": valid_tokens,
                   "valid_label": valid_label,
                   "valid_type_labels": valid_type_labels,
                   "test_tokens": test_tokens,
                   "tokenizer": tokenizer}
        return dataset

    def cal_sample_weights(self):
        # 把 train_df 中的 target 和 所有身份的值变成 bool
        for column in self.identity_list + ["target"]:
            self.train_df[column] = np.where(self.train_df[column] > 0.5, True, False)
        sample_weights = np.ones(len(self.train_df))
        sample_weights += self.train_df["target"]
        if False:
            sample_weights += (~self.train_df["target"]) * self.train_df[self.identity_list].sum(axis=1)
            sample_weights += self.train_df["target"] * (~self.train_df[self.identity_list]).sum(axis=1) * 5
        else:
            sample_weights += (~self.train_df["target"]) * np.where(self.train_df[self.identity_list].sum(axis=1) > 0, 1, 0) * 7
            sample_weights += self.train_df["target"] * np.where((~self.train_df[self.identity_list]).sum(axis=1) > 0, 1, 0) * 7
        sample_weights /= sample_weights.mean()
        # 值留训练集
        sample_weights = sample_weights[:self.train_len]
        return sample_weights

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
        # 输入层
        token_input = Input(shape=(self.max_len,), dtype="int32")
        # 嵌入层
        embedding_layer = Embedding(input_dim=emb_weights.shape[0],
                                    output_dim=emb_weights.shape[1],
                                    weights=[emb_weights],
                                    trainable=True)
        token_emb = embedding_layer(token_input)
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
        dataset = self.create_train_data()
        train_tokens = dataset["train_tokens"]
        train_label = dataset["train_label"]
        train_type_labels = dataset["train_type_labels"]
        valid_tokens = dataset["valid_tokens"]
        valid_label = dataset["valid_label"]
        valid_type_labels = dataset["valid_type_labels"]
        test_tokens = dataset["test_tokens"]
        tokenizer = dataset["tokenizer"]
        sample_weights = self.cal_sample_weights()
        word_embedding = self.create_emb_weights(tokenizer.word_index)
        model = self.build_model(word_embedding)
        for epoch in range(epochs):
            # TODO:先不用test
            model.fit(x=train_tokens,
                      y=[train_label, train_type_labels],
                      batch_size=batch_size,
                      epochs=1,
                      verbose=2,
                      validation_data=([valid_tokens], [valid_label, valid_type_labels]),
                      sample_weight=[sample_weights, np.ones_like(sample_weights)],
                      callbacks=[LearningRateScheduler(lambda _: 1e-3 * (0.6 ** epoch))]
                      )
            # 打分
            y_pred = model.predict(valid_tokens)[0]
            auc_score = self.evaluator.get_final_metric(y_pred) # y_pred 可以是 (n, 1) 也可以是 (n,)  不 squeeze 也没关系。y_true 必须要有正有负，否则无法计算 auc
            if auc_score < previous_auc_score: break
            else: previous_auc_score = auc_score
            print("auc_score:", auc_score)
            if not self.debug_mode:
                model.save(os.path.join(self.data_dir, "model/model[%s]_%d_%.5f" % (self.model_name, epoch, auc_score)))
        # del 训练相关输入和模型，手动清除显存
        training_history = [dataset, train_tokens, train_label, train_type_labels, valid_tokens, valid_label, valid_type_labels, test_tokens, tokenizer, sample_weights, word_embedding, model]
        for training_variable in training_history:
            del training_variable
        K.clear_session()
        gc.collect()


if __name__ == "__main__":
    data_dir = "/Users/hedongfeng/PycharmProjects/unintended_bias/data/"
    trainer = Trainer(data_dir, "model_name", False)
    trainer.train(batch_size=16)
