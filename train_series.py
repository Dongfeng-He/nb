import gc

if __name__ == "__main__":
    data_dir = "/root/nb/data"
    debug_mode = True

    from train.train1_original import Trainer
    trainer = Trainer(data_dir=data_dir, model_name="train1_original", debug_mode=debug_mode)
    trainer.train(epochs=5, batch_size=512)
    del Trainer
    gc.collect()

    from train.train2_binary_label import Trainer
    trainer = Trainer(data_dir=data_dir, model_name="train2_binary_label", debug_mode=debug_mode)
    trainer.train(epochs=5, batch_size=512)
    del Trainer
    gc.collect()

    from train.train3_no_aux import Trainer
    trainer = Trainer(data_dir=data_dir, model_name="train3_no_aux", debug_mode=debug_mode)
    trainer.train(epochs=5, batch_size=512)
    del Trainer
    gc.collect()

    from train.train4_aux_less_weight import Trainer
    trainer = Trainer(data_dir=data_dir, model_name="train4_aux_less_weight", debug_mode=debug_mode)
    trainer.train(epochs=5, batch_size=512)
    del Trainer
    gc.collect()

    from train.train5_no_identity_zero import Trainer
    trainer = Trainer(data_dir=data_dir, model_name="train5_no_identity_zero", debug_mode=debug_mode)
    trainer.train(epochs=5, batch_size=512)
    del Trainer
    gc.collect()

    from train.train6_subgroup_mul_1 import Trainer
    trainer = Trainer(data_dir=data_dir, model_name="train6_subgroup_mul_1", debug_mode=debug_mode)
    trainer.train(epochs=5, batch_size=512)
    del Trainer
    gc.collect()

    from train.train7_subgroup_mul_3 import Trainer
    trainer = Trainer(data_dir=data_dir, model_name="train7_subgroup_mul_3", debug_mode=debug_mode)
    trainer.train(epochs=5, batch_size=512)
    del Trainer
    gc.collect()

    from train.train8_subgroup_mul_5 import Trainer
    trainer = Trainer(data_dir=data_dir, model_name="train8_subgroup_mul_5", debug_mode=debug_mode)
    trainer.train(epochs=5, batch_size=512)
    del Trainer
    gc.collect()

    from train.train9_subgroup_mul_7 import Trainer
    trainer = Trainer(data_dir=data_dir, model_name="train9_subgroup_mul_7", debug_mode=debug_mode)
    trainer.train(epochs=5, batch_size=512)
    del Trainer
    gc.collect()

    from train.train10_len_300 import Trainer
    trainer = Trainer(data_dir=data_dir, model_name="train10_len_300", debug_mode=debug_mode)
    trainer.train(epochs=5, batch_size=512)
    del Trainer
    gc.collect()