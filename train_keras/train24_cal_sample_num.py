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
            self.train_df = pd.read_csv(os.path.join(self.data_dir, "predict.csv"))
            self.test_df = pd.read_csv(os.path.join(self.data_dir, "test.csv"))
        else:
            self.train_df = pd.read_csv(os.path.join(self.data_dir, "predict.csv")).head(1000)
            self.test_df = pd.read_csv(os.path.join(self.data_dir, "test.csv")).head(1000)
        self.train_len = int(len(self.train_df) * self.split_ratio)

    def cal_sample_weights(self, percent):
        train_df = copy.deepcopy(self.train_df.head(int(len(self.train_df) * percent)))
        sample_num_dict = {}
        for column in self.toxicity_type_list + self.identity_list + ["target"]:
            sample_num_dict[column] = np.sum(np.where(train_df[column] > 0.5, 1, 0))
        # 把 train_df 中的 target 和 所有身份的值变成 bool
        for column in self.identity_list + ["target"]:
            train_df[column] = np.where(train_df[column] > 0.5, True, False)
        sample_num_dict["pp"] = np.sum((train_df["target"]) & np.where(train_df[self.identity_list].sum(axis=1) > 0, True, False))
        sample_num_dict["np"] = np.sum((~train_df["target"]) & np.where(train_df[self.identity_list].sum(axis=1) > 0, True, False))
        sample_num_dict["pn"] = np.sum(train_df["target"] & np.where((train_df[self.identity_list]).sum(axis=1) == 0, True, False))
        sample_num_dict["nn"] = np.sum(~train_df["target"] & np.where((train_df[self.identity_list]).sum(axis=1) == 0, True, False))
        for column in self.identity_list:
            sample_num_dict["pp_%s" % column] = np.sum((train_df["target"]) & (train_df[column]))
            sample_num_dict["np_%s" % column] = np.sum((~train_df["target"]) & (train_df[column]))
            sample_num_dict["pn_%s" % column] = np.sum(train_df["target"] & (~train_df[column]))
            sample_num_dict["nn_%s" % column] = np.sum(~train_df["target"] & (~train_df[column]))
            """
            a = ~train_df[column]
            b = np.where(a > 0, 1, 0)
            b = train_df[column]
            c = 1

            a = np.where((~train_df[self.identity_list]).sum(axis=1) == 9, 1, 0)
            b = np.where((train_df[self.identity_list]).sum(axis=1) > 0, 1, 0)
            a = train_df["target"] * train_df[self.identity_list].sum(axis=1)

            a = train_df[self.identity_list + ["target"]]
            a = train_df["target"] * (~train_df[column])
            c = np.where(a > 0, 1, 0)
            b = (train_df["target"]) * (train_df[column])
            print(1)
            """
        print("percent: %f total: %d" % (percent, len(train_df)))
        for key, value in sample_num_dict.items():
            #print(key, value, int((len(train_df) - value) / value))
            print(key, value, int(len(train_df) / value))
        print("")

    def train(self):
        if self.debug_mode: epochs = 1
        #dataset = self.create_dataloader()
        #self.cal_sample_weights(0.2)
        self.cal_sample_weights(0.3)
        #self.cal_sample_weights(0.5)
        self.cal_sample_weights(1)


if __name__ == "__main__":
    data_dir = "/Users/hedongfeng/PycharmProjects/unintended_bias/data/"
    trainer = Trainer(data_dir, "model_name", False)
    trainer.train()

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

    """
    percent: 0.300000 total: 541462
    target 32202 16
    severe_toxicity 1 541462
    obscene 2753 196
    identity_attack 1946 278
    insult 24052 22
    threat 889 609
    male 11809 45
    female 16285 33
    homosexual_gay_or_lesbian 2728 198
    christian 11167 48
    jewish 2221 243
    muslim 4064 133
    black 4129 131
    white 6005 90
    psychiatric_or_mental_illness 1464 369
    pp 5036 107
    np 40053 13
    pn 32202 16
    nn 509260 1
    pp_male 1246 434
    np_male 10563 51
    pn_male 30956 17
    nn_male 498697 1
    pp_female 1668 324
    np_female 14617 37
    pn_female 30534 17
    nn_female 494643 1
    pp_homosexual_gay_or_lesbian 513 1055
    np_homosexual_gay_or_lesbian 2215 244
    pn_homosexual_gay_or_lesbian 31689 17
    nn_homosexual_gay_or_lesbian 507045 1
    pp_christian 549 986
    np_christian 10618 50
    pn_christian 31653 17
    nn_christian 498642 1
    pp_jewish 202 2680
    np_jewish 2019 268
    pn_jewish 32000 16
    nn_jewish 507241 1
    pp_muslim 701 772
    np_muslim 3363 161
    pn_muslim 31501 17
    nn_muslim 505897 1
    pp_black 855 633
    np_black 3274 165
    pn_black 31347 17
    nn_black 505986 1
    pp_white 1163 465
    np_white 4842 111
    pn_white 31039 17
    nn_white 504418 1
    pp_psychiatric_or_mental_illness 197 2748
    np_psychiatric_or_mental_illness 1267 427
    pn_psychiatric_or_mental_illness 32005 16
    nn_psychiatric_or_mental_illness 507993 1
    
    percent: 1.000000 total: 1804874
    target 106438 16
    severe_toxicity 8 225609
    obscene 7648 235
    identity_attack 7633 236
    insult 79887 22
    threat 2793 646
    male 40036 45
    female 50548 35
    homosexual_gay_or_lesbian 10233 176
    christian 35507 50
    jewish 7239 249
    muslim 19666 91
    black 13869 130
    white 23852 75
    psychiatric_or_mental_illness 4077 442
    pp 17777 101
    np 135896 13
    pn 106438 16
    nn 1698436 1
    pp_male 4187 431
    np_male 35849 50
    pn_male 102251 17
    nn_male 1662587 1
    pp_female 4689 384
    np_female 45859 39
    pn_female 101749 17
    nn_female 1652577 1
    pp_homosexual_gay_or_lesbian 2005 900
    np_homosexual_gay_or_lesbian 8228 219
    pn_homosexual_gay_or_lesbian 104433 17
    nn_homosexual_gay_or_lesbian 1690208 1
    pp_christian 2099 859
    np_christian 33408 54
    pn_christian 104339 17
    nn_christian 1665028 1
    pp_jewish 763 2365
    np_jewish 6476 278
    pn_jewish 105675 17
    nn_jewish 1691960 1
    pp_muslim 2974 606
    np_muslim 16692 108
    pn_muslim 103464 17
    nn_muslim 1681744 1
    pp_black 3079 586
    np_black 10790 167
    pn_black 103359 17
    nn_black 1687646 1
    pp_white 4660 387
    np_white 19192 94
    pn_white 101778 17
    nn_white 1679244 1
    pp_psychiatric_or_mental_illness 628 2874
    np_psychiatric_or_mental_illness 3449 523
    pn_psychiatric_or_mental_illness 105810 17
    nn_psychiatric_or_mental_illness 1694987 1
    """