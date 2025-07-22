import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch_optimizer as torch_optim
from torch.utils.data import DataLoader
from datetime import date
from pathlib import Path
import natsort
import random
from src.utilities import *
from src.dataloaders.scalp_eeg_fft_metadata import ScalpEEG_FFT_Metadata_Dataset
from src.models import MV_LSTM, Neural_CNN
from src.config import arg_parse
from src.meter import Meter
from src.training_utils import *
from sklearn import metrics
from sklearn.model_selection import train_test_split
from src.metrics import compute_metric
from src.signal_processing import butter_bandpass_filter_channel_wise_all_channel

class Trainer():
    def __init__(self, args):
        self.experiment_date = date.today().strftime("%y-%m-%d")
        self.experiment_name = f"{self.experiment_date}_{args.experiment_name}"
        self.device = args.device
        self.data_dir = args.data_dir
        self.res_dir = os.path.join(args.work_dir, args.res_dir, "ckpt")
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.seed = args.seed
        self.p_val = args.p_val
        self.p_test = args.p_test
        self.sample_rate = args.sample_rate
        self.window_size = args.window_size
        self.n_channel = args.n_channel
        self.n_hidden = args.n_hidden
        self.n_LSTM_layers = args.n_lstm_layers
        self.linear_layer_dims = eval(args.linear_layer_dims)
        self.pass_band = args.p_band
        self.stop_band = args.s_band
        self.data_type = args.data_type
        os.makedirs(self.res_dir, exist_ok=True)
        self.criterion = nn.BCELoss(reduction="none").to(self.device)

    def get_dataloaders(self, metadata_path, data_dir):
        train_dataset = ScalpEEG_FFT_Metadata_Dataset(
            metadata_path, data_dir, split='train', window_size=self.window_size, n_channels=self.n_channel, sample_rate=self.sample_rate, mode="continuous", pass_band=self.pass_band, stop_band=self.stop_band)
        val_dataset = ScalpEEG_FFT_Metadata_Dataset(
            metadata_path, data_dir, split='val', window_size=self.window_size, n_channels=self.n_channel, sample_rate=self.sample_rate, mode="continuous", pass_band=self.pass_band, stop_band=self.stop_band)
        test_dataset = ScalpEEG_FFT_Metadata_Dataset(
            metadata_path, data_dir, split='test', window_size=self.window_size, n_channels=self.n_channel, sample_rate=self.sample_rate, mode="continuous", pass_band=self.pass_band, stop_band=self.stop_band)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=2, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=2, pin_memory=True, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=2, pin_memory=True, shuffle=True)
        return train_loader, val_loader, test_loader

    def __initialize_model(self):
        return Neural_CNN().to(self.device)

    def __initialize_optimizer(self, model):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.learning_rate)
        return optimizer

    def train(self, train_loader, valid_loader, test_loader, checkpoint_folder, val_log_period=5, epoch_log_period=300, fold_i=0):
        model_instance = self.__initialize_model().float()
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model_instance)
            except Exception as e:
                print(f"torch.compile failed: {e}. Using model without compilation.")
                model = model_instance
        else:
            model = model_instance
        model.train()
        optimizer = self.__initialize_optimizer(model)
        since = time.time()
        best_acc, v_loss, v_acc, v_f1 = -50, 100, 0, 0
        best_model = None
        for epoch in range(self.num_epochs):
            print("-" * 10)
            model.train()
            meter_epoch = Meter()
            for i, sample in enumerate(train_loader, 0):
                waveform = sample["waveform"]
                label = sample["label"]
                label = torch.tensor(label).to(self.device).float()
                input_fft = transform_waveform_to_model_input_gpu(torch.tensor(waveform).to(self.device).float(), self.pass_band, self.stop_band)
                optimizer.zero_grad()
                outputs = model(input_fft).squeeze(1)
                loss = torch.mean(self.criterion(outputs, label))
                meter_epoch.add(
                    output=outputs.detach().cpu().numpy(),
                    label=label.cpu().numpy(),
                    loss=loss.repeat(64).detach().cpu().numpy()
                )
                loss.backward()
                optimizer.step()
                if i % epoch_log_period == 0 and i > 0:
                    print(f"Training: Epoch [{epoch + 1}/{self.num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                    print("outputs", outputs.detach().cpu().numpy())
                    print("labels", label.cpu().numpy())
            loss_epoch = meter_epoch.loss()
            acc_epoch = meter_epoch.accuracy()
            print(f"Epoch {epoch + 1}, loss_s: {loss_epoch:.3f}, accuracy_s: {acc_epoch:.3f}")
            if epoch % val_log_period == 0 and epoch != 0:
                v_loss, v_acc, v_f1, recall, precision = self.validate(valid_loader, model, epoch, True, fold_i)
                if epoch >= 20 and precision > 0.90:
                    best_acc, best_model = pick_best_model_acc(
                        model,
                        best_model,
                        epoch,
                        -v_loss,
                        best_acc,
                        checkpoint_folder,
                        model_name="Neural_FFT_CNN",
                    )
                v_loss, v_acc, v_f1, recall, precision = self.validate(test_loader, model, epoch, False)
        print("Training complete after", epoch, "epochs")
        time_elapsed = time.time() - since
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s ")
        return best_model

    def validate(self, loader, model, epoch_id, dump=False, fold_i=0):
        start = time.time()
        meter = Meter()
        model.eval()
        for i, sample in enumerate(loader, 0):
            with torch.no_grad():
                waveform = sample["waveform"]
                label = sample["label"]
                label = torch.tensor(label).to(self.device).float()
                input_fft = transform_waveform_to_model_input_gpu(torch.tensor(waveform).to(self.device).float(), self.pass_band, self.stop_band)
                outputs = model(input_fft).squeeze(1)
                labels = label.squeeze().cpu()
                outputs = outputs.detach().cpu()
                loss = self.criterion(outputs, labels).repeat(64).numpy()
                meter.add(
                    output=outputs.detach().cpu().numpy(),
                    label=label.cpu().numpy(),
                    loss=loss
                )
        acc = meter.accuracy()
        loss = meter.loss()
        f1, recall, precision = meter.f1()
        print(f"Inference: Time {time.time() - start:.3f}, loss: {loss:.3f}, accuracy: {acc:.3f} , f1: {f1:0.3f}, recall: {recall:0.3f}, precision: {precision:0.3f}")
        if dump:
            folder = f"./logs/{self.experiment_name}/fold{fold_i}"
            if not os.path.exists(folder):
                Path(folder).mkdir(exist_ok=True, parents=True)
            meter.dump_csv(f"./logs/{self.experiment_name}/fold{fold_i}/epoch_{epoch_id}_val.csv")
        return loss, acc, f1, recall, precision

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py -c <config_file> [other options]")
        sys.exit(1)
    mode = sys.argv[1]
    if mode == "-c":
        args = arg_parse(sys.argv[1:])
        print(args)
        np.random.seed(args.seed)
        random.seed(args.seed)
        trainer = Trainer(args)
        train_loader, val_loader, test_loader = trainer.get_dataloaders(args.metadata_path, args.data_dir)
        trainer.train(train_loader, val_loader, test_loader, trainer.res_dir) 