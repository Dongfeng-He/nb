from train import *

if __name__ == "__main__":
    data_dir = "/Users/hedongfeng/PycharmProjects/unintended_bias/data/"
    trainer = Trainer(data_dir=data_dir)
    trainer.train(batch_size=1)