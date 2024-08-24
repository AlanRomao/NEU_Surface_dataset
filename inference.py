import torch
import torch.optim as optim
import torchvision.models as models
import numpy as np
import argparse

from process_data import LoadData
from custom_dataset import CustomImageDataset

from train_pipe import calculate_metrics, calculate_map5
import os
import glob


class Inference():
    def __init__(self, model, data):
        self.model_path = model
        self.data = data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

    def run(self):
        self.model = self.load_model()
        print(self.device)
        self.model.to(self.device)
    
        load_data = LoadData(self.data, 0.0, 0.0, 1.0)
        transform = load_data.get_tranform()

        all_class_dataset = []
        list_all_images_by_class = []
        list_class = [os.path.join(self.data, d) for d in os.listdir(self.data) if os.path.isdir(os.path.join(self.data, d))]
        for each_class in list_class:
            class_name = each_class.split("\\")[-1]
            if class_name not in all_class_dataset:
                all_class_dataset.append(class_name)
                list_all_images_by_class.append([])

            idx_classe = all_class_dataset.index(class_name)
            
            files = glob.glob(f'{each_class}\\' + '*.jpg')
            list_all_images_by_class[idx_classe] += files

        load_data.run_split_data(list_all_images_by_class)
        test_dataset = CustomImageDataset(image_paths=load_data.list_all_test_imgs, labels=load_data.list_all_test_labels, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

        all_labels = []
        all_preds = []

        self.model.eval()  

        with torch.no_grad():  
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        precision, recall, f1 = calculate_metrics(all_preds, all_labels)
        map5 = calculate_map5(all_labels, all_preds)
        print(f'TEST - Precision: {precision:.4f} - Recall: {recall:.4f} - F1 Score: {f1:.4f} - mAP5: {map5:.4f}')

    def load_model(self, path='model.pth'):
        model = torch.load(path, map_location=self.device)
        model.eval()
        return model, None

    def load_model(self):
        model = models.resnet18(pretrained=False, num_classes=6)

        checkpoint = torch.load(self.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

###########################################################################################################################################################
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Please insert the model path..")
    parser.add_argument("--data", default=None, help="Please insert the dataset path..")    
    opt = parser.parse_args()
    return opt

###########################################################################################################################################################
if __name__ == "__main__":
    opt = parse_opt()
    inference = Inference(opt.model, opt.data)
    inference.run()