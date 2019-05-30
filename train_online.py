from train import *

if __name__ == "__main__":
    data_dir = "/root/nb/data"
    trainer = Trainer(data_dir=data_dir)
    trainer.train(epochs=5, batch_size=512)