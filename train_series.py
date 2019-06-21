import gc
import time

if __name__ == "__main__":
    data_dir = "/root/nb/data"
    debug_mode = False
    """
    from train_keras.train1_original import Trainer
    print("start to train_keras: train1_original")
    trainer = Trainer(data_dir=data_dir, model_name="train1_original", debug_mode=debug_mode)
    trainer.train_keras(epochs=5, batch_size=512)

    from train_keras.train2_binary_label import Trainer
    print("start to train_keras: train2_binary_label")
    trainer = Trainer(data_dir=data_dir, model_name="train2_binary_label", debug_mode=debug_mode)
    trainer.train_keras(epochs=5, batch_size=512)

    from train_keras.train3_no_aux import Trainer
    print("start to train_keras: train3_no_aux")
    trainer = Trainer(data_dir=data_dir, model_name="train3_no_aux", debug_mode=debug_mode)
    trainer.train_keras(epochs=5, batch_size=512)

    from train_keras.train4_aux_less_weight import Trainer
    print("start to train_keras: train4_aux_less_weight")
    trainer = Trainer(data_dir=data_dir, model_name="train4_aux_less_weight", debug_mode=debug_mode)
    trainer.train_keras(epochs=5, batch_size=512)

    from train_keras.train5_no_identity_zero import Trainer
    print("start to train_keras: train5_no_identity_zero")
    trainer = Trainer(data_dir=data_dir, model_name="train5_no_identity_zero", debug_mode=debug_mode)
    trainer.train_keras(epochs=5, batch_size=512)

    from train_keras.train6_subgroup_mul_1 import Trainer
    print("start to train_keras: train6_subgroup_mul_1")
    trainer = Trainer(data_dir=data_dir, model_name="train6_subgroup_mul_1", debug_mode=debug_mode)
    trainer.train_keras(epochs=5, batch_size=512)

    from train_keras.train7_subgroup_mul_3 import Trainer
    print("start to train_keras: train7_subgroup_mul_3")
    trainer = Trainer(data_dir=data_dir, model_name="train7_subgroup_mul_3", debug_mode=debug_mode)
    trainer.train_keras(epochs=5, batch_size=512)

    from train_keras.train8_subgroup_mul_5 import Trainer
    print("start to train_keras: train8_subgroup_mul_5")
    trainer = Trainer(data_dir=data_dir, model_name="train8_subgroup_mul_5", debug_mode=debug_mode)
    trainer.train_keras(epochs=5, batch_size=512)

    from train_keras.train9_subgroup_mul_7 import Trainer
    print("start to train_keras: train9_subgroup_mul_7")
    trainer = Trainer(data_dir=data_dir, model_name="train9_subgroup_mul_7", debug_mode=debug_mode)
    trainer.train_keras(epochs=5, batch_size=512)

    from train_keras.train10_len_300 import Trainer
    print("start to train_keras: train10_len_300")
    trainer = Trainer(data_dir=data_dir, model_name="train10_len_300", debug_mode=debug_mode)
    trainer.train_keras(epochs=5, batch_size=512)

    from train_keras.train11_identity_original import Trainer
    print("start to train_keras: train11_identity_original")
    trainer = Trainer(data_dir=data_dir, model_name="train11_identity_original", debug_mode=debug_mode)
    trainer.train_keras(epochs=5, batch_size=512)

    from train_keras.train12_identity_binary import Trainer
    print("start to train_keras: train12_identity_binary")
    trainer = Trainer(data_dir=data_dir, model_name="train12_identity_binary", debug_mode=debug_mode)
    trainer.train_keras(epochs=5, batch_size=512)

    from train_keras.train13_identity_original_sum import Trainer
    print("start to train_keras: train13_identity_original_sum")
    trainer = Trainer(data_dir=data_dir, model_name="train13_identity_original_sum", debug_mode=debug_mode)
    trainer.train_keras(epochs=5, batch_size=512)

    from train_keras.train14_identity_binary_or import Trainer
    print("start to train_keras: train14_identity_binary_or")
    trainer = Trainer(data_dir=data_dir, model_name="train14_identity_binary_or", debug_mode=debug_mode)
    trainer.train_keras(epochs=5, batch_size=512)

    from train_keras.train23_aux_identity_2_layer_hidden import Trainer
    print("start to train_keras: train23_aux_identity_2_layer_hidden")
    trainer = Trainer(data_dir=data_dir, model_name="train23_aux_identity_2_layer_hidden", debug_mode=debug_mode)
    trainer.train_keras(epochs=5, batch_size=512)

    from train_keras.train20_identity_2_layer_hidden import Trainer
    print("start to train_keras: train20_identity_2_layer_hidden")
    trainer = Trainer(data_dir=data_dir, model_name="train20_identity_2_layer_hidden", debug_mode=debug_mode)
    trainer.train_keras(epochs=5, batch_size=512)

    from train_keras.train15_identity_original_mul_5 import Trainer
    print("start to train_keras: train15_identity_original_mul_5")
    trainer = Trainer(data_dir=data_dir, model_name="train15_identity_original_mul_5", debug_mode=debug_mode)
    trainer.train_keras(epochs=5, batch_size=512)

    from train_keras.train16_identity_1_layer_hidden import Trainer
    print("start to train_keras: train16_identity_1_layer_hidden")
    trainer = Trainer(data_dir=data_dir, model_name="train16_identity_1_layer_hidden", debug_mode=debug_mode)
    trainer.train_keras(epochs=5, batch_size=512)

    from train_keras.train17_identity_1_layer_hidden_feedback import Trainer
    print("start to train_keras: train17_identity_1_layer_hidden_feedback")
    trainer = Trainer(data_dir=data_dir, model_name="train17_identity_1_layer_hidden_feedback", debug_mode=debug_mode)
    trainer.train_keras(epochs=5, batch_size=512)

    from train_keras.train18_aux_identity_1_layer_hidden import Trainer
    print("start to train_keras: train18_aux_and_identity_1_layer_hidden")
    trainer = Trainer(data_dir=data_dir, model_name="train18_aux_and_identity_1_layer_hidden", debug_mode=debug_mode)
    trainer.train_keras(epochs=5, batch_size=512)

    from train_keras.train19_identity_same_layer_hidden import Trainer
    print("start to train_keras: train19_identity_same_layer_hidden")
    trainer = Trainer(data_dir=data_dir, model_name="train19_identity_same_layer_hidden", debug_mode=debug_mode)
    trainer.train_keras(epochs=5, batch_size=512)

    from train_keras.train22_identity_2_layer_hidden2 import Trainer
    print("start to train_keras: train22_identity_2_layer_hidden2")
    trainer = Trainer(data_dir=data_dir, model_name="train22_identity_2_layer_hidden2", debug_mode=debug_mode)
    trainer.train_keras(epochs=5, batch_size=512)

    from train_keras.train21_focal_loss import Trainer
    print("start to train_keras: train21_focal_loss")
    trainer = Trainer(data_dir=data_dir, model_name="train21_focal_loss", debug_mode=debug_mode)
    trainer.predict(epochs=5, batch_size=512)

    from train_pytorch.train1_original import Trainer
    print("start to train_pytorch: train1_original")
    trainer = Trainer(data_dir=data_dir, model_name="train1_original_part", epochs=10, batch_size=512, part=0.3, debug_mode=debug_mode)
    trainer.predict()

    from train_pytorch.train9_focal_loss_seed_total import Trainer
    print("start to train_pytorch: train9_focal_loss_seed")
    trainer = Trainer(data_dir=data_dir, model_name="train9_focal_loss_seed", epochs=30, batch_size=512, part=1., seed=1234, debug_mode=debug_mode)
    trainer.predict()
    del trainer
    gc.collect()
    
    from train_bert.train1_aux_identity_gate import Trainer
    print("start to train_bert: train1_original")
    trainer = Trainer(data_dir=data_dir, model_name="train1_original", epochs=3, batch_size=64, base_batch_size=32, part=1., seed=1234, debug_mode=debug_mode)
    trainer.predict()
    del trainer
    gc.collect()
    """
    from train_bert.train4_aux_identity import Trainer
    print("start to train_bert: train4_aux_identity")
    trainer = Trainer(data_dir=data_dir, model_name="train4_aux_identity", epochs=3, batch_size=64, base_batch_size=32, part=1., seed=1234, debug_mode=debug_mode)
    trainer.train()
    del trainer
    gc.collect()