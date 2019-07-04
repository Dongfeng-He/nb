import os
from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch

model_path = "/root/nb/data/nl2sql_data/chinese_L-12_H-768_A-12/"

if os.path.exists(model_path + "pytorch_model.bin") is False:
    convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
        model_path + 'bert_model.ckpt',
        model_path + 'bert_config.json',
        model_path + 'pytorch_model.bin')

"""
get /root/nb/data/nl2sql_data/chinese_L-12_H-768_A-12/pytorch_model.bin /Users/hedongfeng/Desktop/
put /Users/hedongfeng/PycharmProjects/unintended_bias/data/nl2sql_data.zip  /root/nb/data/
"""
