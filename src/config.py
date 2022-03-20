import torch

CONFIG = {"seed": 2022,
          "epochs": 50,
          "img_width": 512,
          "img_height": 512,
          "model_name": "tf_efficientnet_b7_ns",
          "num_classes": 15587,
          "embedding_size": 512,
          "train_batch_size": 16,
          "valid_batch_size": 64,
          "learning_rate": 1e-3,
          "scheduler": 'CosineAnnealingLR',
          "min_lr": 1e-6,
          "T_max": 500,
          "weight_decay": 1e-6,
          "n_fold": 5,
          "n_accumulate": 1,
          "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
          # ArcFace Hyperparameters
          "s": 30.0,
          "m": 0.50,
          "ls_eps": 0.0,
          "easy_margin": False,
          "checkpoint_path": '/content/drive/MyDrive/HappyWhale/embed_models/exp1',
          }