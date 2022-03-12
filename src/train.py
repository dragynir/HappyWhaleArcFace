import wandb
import torch
import time
import copy
from collections import defaultdict
import gc
import numpy as np
from config import CONFIG
import torch.nn as nn
from tqdm import tqdm
from torch.optim import lr_scheduler
import torch.optim as optim
from colorama import Fore, Back, Style
import pandas as pd

from models import HappyWhaleModel
from data.loaders import prepare_loaders


sr_ = Style.RESET_ALL
b_ = Fore.BLUE

class Runner(object):

    def __init__(self, df):

        self.model = HappyWhaleModel(CONFIG['model_name'], CONFIG['embedding_size'])
        self.model.to(CONFIG['device'])
        self.train_loader, self.valid_loader = prepare_loaders(df, fold=0)

        self.criterion = self.create_criterion()
        self.optimzer, self.scheduler = self.create_optimizer()


    def run_training(self, num_epochs):


        run = wandb.init(project='HappyWhale', 
                 config=CONFIG,
                 job_type='Train',
                 tags=['arcface', 'gem-pooling', 'effnet-b0-ns', '448'],
                 anonymous='must')

        # To automatically log gradients
        wandb.watch(self.model, log_freq=100)

        if torch.cuda.is_available():
            print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

        start = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_epoch_loss = np.inf
        history = defaultdict(list)

        for epoch in range(1, num_epochs + 1):
            gc.collect()
            train_epoch_loss = self.train_one_epoch(device=CONFIG['device'], epoch=epoch)

            val_epoch_loss = self.valid_one_epoch(device=CONFIG['device'],
                                             epoch=epoch)

            history['Train Loss'].append(train_epoch_loss)
            history['Valid Loss'].append(val_epoch_loss)

            # Log the metrics
            wandb.log({"Train Loss": train_epoch_loss})
            wandb.log({"Valid Loss": val_epoch_loss})

            # deep copy the model
            if val_epoch_loss <= best_epoch_loss:
                print(f"{b_}Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})")
                best_epoch_loss = val_epoch_loss
                run.summary["Best Loss"] = best_epoch_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                PATH = "Loss{:.4f}_epoch{:.0f}.bin".format(best_epoch_loss, epoch)
                torch.save(self.model.state_dict(), PATH)
                # Save a model file from the current directory
                print(f"Model Saved{sr_}")

            print()

        end = time.time()
        time_elapsed = end - start
        print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
        print("Best Loss: {:.4f}".format(best_epoch_loss))

        # load best model weights
        self.model.load_state_dict(best_model_wts)

        return self.model, history

    def create_criterion(self):
        return nn.CrossEntropyLoss()
    
    def create_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=CONFIG['learning_rate'], 
                            weight_decay=CONFIG['weight_decay'])
        scheduler = self.fetch_scheduler(optimizer)  

        return optimizer, scheduler
    
    def fetch_scheduler(self, optimizer):
        if CONFIG['scheduler'] == 'CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['T_max'], 
                                                    eta_min=CONFIG['min_lr'])
        elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CONFIG['T_0'], 
                                                                eta_min=CONFIG['min_lr'])
        elif CONFIG['scheduler'] == None:
            return None
            
        return scheduler

    def train_one_epoch(self, device, epoch):
        self.model.train()

        dataset_size = 0
        running_loss = 0.0

        bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for step, data in bar:
            images = data['image'].to(device, dtype=torch.float)
            labels = data['label'].to(device, dtype=torch.long)

            batch_size = images.size(0)

            outputs = self.model(images, labels)
            loss = self.criterion(outputs, labels)
            loss = loss / CONFIG['n_accumulate']

            loss.backward()

            if (step + 1) % CONFIG['n_accumulate'] == 0:
                self.optimizer.step()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

            running_loss += (loss.item() * batch_size)
            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size

            bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                            LR=self.optimizer.param_groups[0]['lr'])
        gc.collect()

        return epoch_loss

    @torch.inference_mode()
    def valid_one_epoch(self, device, epoch):
        self.model.eval()

        dataset_size = 0
        running_loss = 0.0

        bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))
        for step, data in bar:
            images = data['image'].to(device, dtype=torch.float)
            labels = data['label'].to(device, dtype=torch.long)

            batch_size = images.size(0)

            outputs = self.model(images, labels)
            loss = self.criterion(outputs, labels)

            running_loss += (loss.item() * batch_size)
            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size

            bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
                            LR=self.optimizer.param_groups[0]['lr'])

        gc.collect()

        return epoch_loss

if __name__ == '__main__':
    dataset_path = '/content/dataset.csv'
    df = pd.read_csv(dataset_path)
    runner = Runner(df)
    runner.run_training(CONFIG['epochs'])