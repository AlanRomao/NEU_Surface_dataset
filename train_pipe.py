import time
import copy
import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torchvision.models import ResNet18_Weights

from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

from process_data import LoadData


class TrainPipe():
    def __init__(self, main_directory, train = 0.7, val = 0.2 , test = 0.1, num_epochs = 10):
        self.main_path = main_directory
        self.split_train_value = train
        self.split_test_value = test
        self.split_val_value = val
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs

    def run(self):
        load_data = LoadData(self.main_path, self.split_train_value, self.split_val_value, self.split_test_value)
        load_data.run()

        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(load_data.all_class_dataset))  
        model = model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        dataloaders = {'train': load_data.train_loader, 'val': load_data.val_loader}
        model = self.train_model(model, dataloaders, criterion, optimizer, num_epochs=self.num_epochs)

        self.save_model(model, optimizer)

        model.eval()
        all_labels = []
        all_preds = []
        
        with torch.no_grad():  
            for inputs, labels in load_data.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        precision, recall, f1 = calculate_metrics(all_preds, all_labels)
        map5 = calculate_map5(all_labels, all_preds)
        output_csv_path = 'metrics.csv'
        self.create_csv_result(output_csv_path, precision, recall, f1, map5)
        print(f'TEST - Precision: {precision:.4f} - Recall: {recall:.4f} - F1 Score: {f1:.4f} - mAP5: {map5:.4f}')



    def train_model(self, model, dataloaders, criterion, optimizer, num_epochs=10):
        best_model_wts = model.state_dict()
        best_acc = 0.0
        print('--- Starting training ---')
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            
            for phase in ['train', 'val']:
                model.train() if phase == 'train' else model.eval()
                running_loss = 0.0
                all_preds = []
                all_labels = []

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                precision, recall, f1 = calculate_metrics(all_preds, all_labels)
                map5 = calculate_map5(all_labels, all_preds)
                
                print(f'{phase} Loss: {epoch_loss:.4f}- Precision: {precision:.4f} - Recall: {recall:.4f} - F1 Score: {f1:.4f} - mAP5: {map5:.4f}')
                
                if phase == 'val' and f1 > best_acc:
                    best_acc = f1
                    best_model_wts = model.state_dict()

            print('-' * 20)

        model.load_state_dict(best_model_wts)
        return model

    def save_model(self, model, optimizer, path='model.pth'):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)

    def create_csv_result(self, output_csv_path, precision, recall, f1, map5):
        metrics = {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'mAP5': map5
        }

        df = pd.DataFrame([metrics])
        df.to_csv(output_csv_path, index=False)

def calculate_metrics(preds, labels, average='macro'):
    precision = precision_score(labels, preds, average=average)
    recall = recall_score(labels, preds, average=average)
    f1 = f1_score(labels, preds, average=average)
    return precision, recall, f1

def calculate_map5(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    if len(y_true.shape) == 1:
        y_true = np.expand_dims(y_true, axis=1)
    if len(y_score.shape) == 1:
        y_score = np.expand_dims(y_score, axis=1)

    map5 = average_precision_score(y_true, y_score, average='macro')
    return map5
    
###########################################################################################################################################################
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="Please insert the dataset path..")
    parser.add_argument("--train", type=float, default=0.7, help="Please enter the values ​​for train split")
    parser.add_argument("--val", type=float, default=0.2, help="Please enter the values ​​for val split")
    parser.add_argument("--test", type=float, default=0.1, help="Please enter the values ​​for test split")
    parser.add_argument("--epoch", type=int, default=10, help="Please enter the values ​​for test split")
    
    opt = parser.parse_args()
    return opt

###########################################################################################################################################################
if __name__ == "__main__":
    opt = parse_opt()
    train_pipe = TrainPipe(opt.dataset)
    train_pipe.run()

