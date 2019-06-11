import gc
import time

if __name__ == "__main__":
    data_dir = "/root/nb/data"
    debug_mode = False
    from train_pytorch.inference import Trainer
    trainer = Trainer(data_dir, debug_mode=False)
    trainer.eval()
