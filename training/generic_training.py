import sys
import traceback
from datetime import datetime
import torch
import torchmetrics
import yaml
from torch import nn
# import data loader
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from data import constants
from data.custom_augmentations import get_augmentation_transform
from data.input_data import ImageSegmentationDataset, ImageSegmentationMultipleSlicesAsChannelsDataset, ImageSegmentationLSTM
from model.models import get_model_from_name, Autoencoder
from model_metrics.metrics import get_criterion_from_name


def generic_training(config=None):
    with wandb.init(config=config):
        config = wandb.config
        trainer = TrainingClass(config)
        trainer.train_unet()
        #trainer.train_hybrid()


class TrainingClass:
    def __init__(self, config):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.batch_size = config.batch_size
        self.lr = config.learning_rate
        self.num_epochs = config.num_epochs
        self.criterion_name = config.criterion
        self.model_name = config.model
        self.training_path = config.training_path
        self.validation_path = config.validation_path
        self.max_non_improvement_epochs = config.max_non_improvement_epochs
        self.IoU = torchmetrics.JaccardIndex(num_classes=1, task='binary').to(self.device)
        self.min_val_iou_for_saving = config.min_val_iou_for_saving
        self.dataset_as_slices = config.dataset_as_slices
        self.num_slices = config.num_slices
        self.use_lstm = config.use_lstm

        self.model = get_model_from_name(self.model_name, self.num_slices).to(self.device)
        self.criterion = get_criterion_from_name(self.criterion_name)
        # for now adam is fixed, but we can add more optimizers later
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        transform_train = get_augmentation_transform(constants.MODE_AUGMENTATION_TRAIN)
        transform_val = get_augmentation_transform(constants.MODE_AUGMENTATION_VAL)
        limit_for_testing = None
        starting_index = 0

        if config.debug_testing:
            self.validation_path = self.training_path
            starting_index = 0
            limit_for_testing = 10
            transform_train = transform_val

        if self.use_lstm:
            self.training_dataset = ImageSegmentationLSTM(csv_file=self.training_path,
                                                          limit_for_testing=limit_for_testing,
                                                            starting_index=starting_index,
                                                            transform=transform_train,
                                                          slices=self.num_slices)
            self.validation_dataset = ImageSegmentationLSTM(csv_file=self.validation_path,
                                                            limit_for_testing=limit_for_testing,
                                                            starting_index=starting_index,
                                                            transform=transform_val,
                                                            slices=self.num_slices)
        else:
            # if  self.dataset_slices is False
            if not self.dataset_as_slices:
                self.training_dataset = ImageSegmentationDataset(csv_file=self.training_path,
                                                                 limit_for_testing=limit_for_testing,
                                                                    starting_index=starting_index,
                                                                 transform=transform_train)
                self.validation_dataset = ImageSegmentationDataset(csv_file=self.validation_path,
                                                                   limit_for_testing=limit_for_testing,
                                                                    starting_index=starting_index,
                                                                   transform=transform_val)
            else:
                self.training_dataset = ImageSegmentationMultipleSlicesAsChannelsDataset(csv_file=self.training_path,
                                                                                         slices=self.num_slices,
                                                                                         limit_for_testing=limit_for_testing,
                                                                                         starting_index=starting_index,
                                                                                         transform=transform_train)

                self.validation_dataset = ImageSegmentationMultipleSlicesAsChannelsDataset(csv_file=self.validation_path,
                                                                                           slices=self.num_slices,
                                                                                           limit_for_testing=limit_for_testing,
                                                                                           starting_index=starting_index,
                                                                                           transform=transform_val)

        self.train_loader = DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False)

    def get_input_and_target_for_different_models(self, images, masks):
        if self.model_name == constants.UNET_MODEL \
                or self.model_name == constants.UNET_MODEL_GENERIC \
                or self.model_name == constants.UNET_LSTM:
            return images, masks
        elif self.model_name == constants.AUTOENCODER_MODEL:
            return masks, masks

    def train_unet(self):
        best_val_iou = 0.0
        best_val_loss = 1000.0
        number_of_epochs_without_improvement = 0
        for epoch in tqdm(range(self.num_epochs)):
            self.model.train()
            total_train_loss = 0.0
            total_train_iou = 0.0
            # add twdm progress bar
            for i, (images, masks, name, sn) in tqdm(enumerate(self.train_loader)):
                self.optimizer.zero_grad()
                images = images.to(self.device)
                masks = masks.to(self.device)
                images, masks = self.get_input_and_target_for_different_models(images, masks)
                # forward pass
                outputs = self.model(images)
                outputs = torch.sigmoid(outputs)
                # calculate loss
                loss = self.criterion(outputs, masks)
                # backward pass
                loss.backward()
                self.optimizer.step()
                # compute metrics
                single_batch_train_iou = self.IoU(outputs, masks)
                total_train_iou += single_batch_train_iou
                total_train_loss += loss.item()

            # compute loggable metrics
            mean_train_iou = total_train_iou / len(self.train_loader)
            mean_train_loss = total_train_loss / len(self.train_loader)

            '''wandb.log({"train_loss": mean_train_loss,
                       "train_iou": mean_train_iou})'''

            with torch.no_grad():
                val_loss = 0.0
                self.model.eval()
                total_val_iou = 0.0
                total_val_loss = 0.0
                for i, (images, masks, name, sn) in enumerate(self.val_loader):
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    images, masks = self.get_input_and_target_for_different_models(images, masks)
                    outputs = self.model(images)
                    outputs = torch.sigmoid(outputs)
                    loss = self.criterion(outputs, masks)
                    single_batch_val_iou = self.IoU(outputs, masks)
                    total_val_iou += single_batch_val_iou
                    total_val_loss += loss.item()
            # compute loggable metrics
            mean_val_iou = total_val_iou / len(self.val_loader)
            mean_val_loss = total_val_loss / len(self.val_loader)

            # log all loggable metrics
            wandb.log({"train_loss": mean_train_loss,
                       "train_iou": mean_train_iou,
                       "val_loss": mean_val_loss,
                       "val_iou": mean_val_iou})

            number_of_epochs_without_improvement += 1
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                number_of_epochs_without_improvement = 0
                if best_val_loss < self.min_val_iou_for_saving:
                    torch.save(self.model.state_dict(),
                               f"model_unet_generic_{mean_val_loss}{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}.pth")
            else:
                if number_of_epochs_without_improvement > self.max_non_improvement_epochs:
                    print("Early stopping")
                    break

    def train_hybrid(self):
        best_val_iou = 0.0
        best_val_loss = 1000.0
        number_of_epochs_without_improvement = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        autoencoder = Autoencoder().to(device)
        autoencoder.load_state_dict(
            torch.load('../training/model_0.83041048049926762023_02_18__12_51_37.pth', map_location=torch.device('cpu')))
        autoencoder.eval()
        encoder = autoencoder.encoder
        ddl = torch.nn.Sequential(encoder, *list(autoencoder.flatten.children())[:2])
        ddl.eval()

        for epoch in tqdm(range(self.num_epochs)):
            self.model.train()
            total_train_loss = 0.0
            total_train_iou = 0.0
            # add twdm progress bar
            for i, (images, masks, name, sn) in tqdm(enumerate(self.train_loader)):
                self.optimizer.zero_grad()
                images = images.to(self.device)
                masks = masks.to(self.device)
                images, masks = self.get_input_and_target_for_different_models(images, masks)
                # forward pass
                outputs = self.model(images)
                outputs = torch.sigmoid(outputs)
                # calculate loss
                loss = self.criterion(outputs, masks)

                # ddl loss
                ls_gt = ddl(masks)
                ls_pred = ddl(outputs)
                # sigmoid
                ls_gt = nn.Sigmoid()(ls_gt)
                ls_pred = nn.Sigmoid()(ls_pred)
                loss_bce = nn.BCELoss()(ls_pred, ls_gt)

                loss = loss + 0.1 * loss_bce

                # backward pass
                loss.backward()
                self.optimizer.step()
                # compute metrics
                single_batch_train_iou = self.IoU(outputs, masks)
                total_train_iou += single_batch_train_iou
                total_train_loss += loss.item()

            # compute loggable metrics
            mean_train_iou = total_train_iou / len(self.train_loader)
            mean_train_loss = total_train_loss / len(self.train_loader)

            '''wandb.log({"train_loss": mean_train_loss,
                       "train_iou": mean_train_iou})'''

            with torch.no_grad():
                val_loss = 0.0
                self.model.eval()
                total_val_iou = 0.0
                total_val_loss = 0.0
                for i, (images, masks, name, sn) in enumerate(self.val_loader):
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    images, masks = self.get_input_and_target_for_different_models(images, masks)
                    outputs = self.model(images)
                    outputs = torch.sigmoid(outputs)
                    loss = self.criterion(outputs, masks)

                    # ddl loss
                    ls_gt = ddl(masks)
                    ls_pred = ddl(outputs)
                    # sigmoid
                    ls_gt = nn.Sigmoid()(ls_gt)
                    ls_pred = nn.Sigmoid()(ls_pred)
                    loss_bce = nn.BCELoss()(ls_pred, ls_gt)

                    loss = loss + 0.1 * loss_bce


                    single_batch_val_iou = self.IoU(outputs, masks)
                    total_val_iou += single_batch_val_iou
                    total_val_loss += loss.item()
            # compute loggable metrics
            mean_val_iou = total_val_iou / len(self.val_loader)
            mean_val_loss = total_val_loss / len(self.val_loader)

            # log all loggable metrics
            wandb.log({"train_loss": mean_train_loss,
                       "train_iou": mean_train_iou,
                       "val_loss": mean_val_loss,
                       "val_iou": mean_val_iou})

            number_of_epochs_without_improvement += 1
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                number_of_epochs_without_improvement = 0
                if mean_val_loss < self.min_val_iou_for_saving:
                    torch.save(self.model.state_dict(),
                               f"model_unet_yaml_{mean_val_loss}{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}.pth")
            else:
                if number_of_epochs_without_improvement > self.max_non_improvement_epochs:
                    print("Early stopping")
                    break


def main():
    wandb.login()
    with open("yaml_files/unet-lstm.yaml", "r") as f:
        config = yaml.safe_load(f)

    sweep_id = wandb.sweep(config, project="unet-lstm-fix-3")
    wandb.agent(sweep_id, generic_training, count=10)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)
