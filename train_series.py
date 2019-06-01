import gc
import time

if __name__ == "__main__":
    data_dir = "/root/nb/data"
    debug_mode = False
    """
    from train.train1_original import Trainer
    print("start to train: train1_original")
    trainer = Trainer(data_dir=data_dir, model_name="train1_original", debug_mode=debug_mode)
    trainer.train(epochs=5, batch_size=512)

    from train.train2_binary_label import Trainer
    print("start to train: train2_binary_label")
    trainer = Trainer(data_dir=data_dir, model_name="train2_binary_label", debug_mode=debug_mode)
    trainer.train(epochs=5, batch_size=512)

    from train.train3_no_aux import Trainer
    print("start to train: train3_no_aux")
    trainer = Trainer(data_dir=data_dir, model_name="train3_no_aux", debug_mode=debug_mode)
    trainer.train(epochs=5, batch_size=512)

    from train.train4_aux_less_weight import Trainer
    print("start to train: train4_aux_less_weight")
    trainer = Trainer(data_dir=data_dir, model_name="train4_aux_less_weight", debug_mode=debug_mode)
    trainer.train(epochs=5, batch_size=512)

    from train.train5_no_identity_zero import Trainer
    print("start to train: train5_no_identity_zero")
    trainer = Trainer(data_dir=data_dir, model_name="train5_no_identity_zero", debug_mode=debug_mode)
    trainer.train(epochs=5, batch_size=512)

    from train.train6_subgroup_mul_1 import Trainer
    print("start to train: train6_subgroup_mul_1")
    trainer = Trainer(data_dir=data_dir, model_name="train6_subgroup_mul_1", debug_mode=debug_mode)
    trainer.train(epochs=5, batch_size=512)

    from train.train7_subgroup_mul_3 import Trainer
    print("start to train: train7_subgroup_mul_3")
    trainer = Trainer(data_dir=data_dir, model_name="train7_subgroup_mul_3", debug_mode=debug_mode)
    trainer.train(epochs=5, batch_size=512)

    from train.train8_subgroup_mul_5 import Trainer
    print("start to train: train8_subgroup_mul_5")
    trainer = Trainer(data_dir=data_dir, model_name="train8_subgroup_mul_5", debug_mode=debug_mode)
    trainer.train(epochs=5, batch_size=512)
    """
    from train.train9_subgroup_mul_7 import Trainer
    print("start to train: train9_subgroup_mul_7")
    trainer = Trainer(data_dir=data_dir, model_name="train9_subgroup_mul_7", debug_mode=debug_mode)
    trainer.train(epochs=5, batch_size=512)

    from train.train10_len_300 import Trainer
    print("start to train: train10_len_300")
    trainer = Trainer(data_dir=data_dir, model_name="train10_len_300", debug_mode=debug_mode)
    trainer.train(epochs=5, batch_size=512)

    from train.train11_identity_original import Trainer
    print("start to train: train11_identity_original")
    trainer = Trainer(data_dir=data_dir, model_name="train11_identity_original", debug_mode=debug_mode)
    trainer.train(epochs=5, batch_size=512)

    from train.train12_identity_binary import Trainer
    print("start to train: train12_identity_binary")
    trainer = Trainer(data_dir=data_dir, model_name="train12_identity_binary", debug_mode=debug_mode)
    trainer.train(epochs=5, batch_size=512)

    from train.train13_identity_original_sum import Trainer
    print("start to train: train13_identity_original_sum")
    trainer = Trainer(data_dir=data_dir, model_name="train13_identity_original_sum", debug_mode=debug_mode)
    trainer.train(epochs=5, batch_size=512)

    from train.train14_identity_binary_or import Trainer
    print("start to train: train14_identity_binary_or")
    trainer = Trainer(data_dir=data_dir, model_name="train14_identity_binary_or", debug_mode=debug_mode)
    trainer.train(epochs=5, batch_size=512)