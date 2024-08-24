import glob
import os
import random

from torch.utils.data import DataLoader
from torchvision import transforms

from custom_dataset import CustomImageDataset

class LoadData():
    def __init__(self, main_directory, train = 0.7, val = 0.2 , test = 0.1):
        self.main_path = main_directory
        self.split_train_value = train
        self.split_test_value = test
        self.split_val_value = val

        self.list_all_train_imgs = []
        self.list_all_train_labels = []
        self.list_all_test_imgs = []
        self.list_all_test_labels = []
        self.list_all_val_imgs = []
        self.list_all_val_labels = []

        self.train_loader = []
        self.test_loader = []
        self.val_loader = []

        self.all_class_dataset = []

    def run(self):
        split_datasets = [os.path.join(self.main_path, d) for d in os.listdir(self.main_path) if os.path.isdir(os.path.join(self.main_path, d))]

        
        list_all_images_by_class = []

        for div_dataset in split_datasets:
            annotation_imagens = [os.path.join(div_dataset, d) for d in os.listdir(div_dataset) if os.path.isdir(os.path.join(div_dataset, d))]
            for image_path in annotation_imagens:
                if 'images' in image_path: 
                    list_class = [os.path.join(image_path, d) for d in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, d))]
                    for each_class in list_class:
                        class_name = each_class.split("\\")[-1]
                        if class_name not in self.all_class_dataset:
                            self.all_class_dataset.append(class_name)
                            list_all_images_by_class.append([])

                        idx_classe = self.all_class_dataset.index(class_name)
                        
                        files = glob.glob(f'{each_class}\\' + '*.jpg')
                        list_all_images_by_class[idx_classe] += files
        
        self.run_split_data(list_all_images_by_class)


        train_transform = self.get_tranform(True)
        transform = self.get_tranform()

        train_dataset = CustomImageDataset(image_paths=self.list_all_train_imgs, labels=self.list_all_train_labels, transform=train_transform)
        val_dataset = CustomImageDataset(image_paths=self.list_all_val_imgs, labels=self.list_all_val_labels, transform=transform)
        test_dataset = CustomImageDataset(image_paths=self.list_all_test_imgs, labels=self.list_all_test_labels, transform=transform)

        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


    def run_split_data(self, list_all_images_by_class):
        for idx_class, img_path in enumerate(list_all_images_by_class):
            random.shuffle(img_path)
            total = len(img_path)
            len_train = int(total * self.split_train_value)
            len_test = int(total * self.split_test_value)
            len_val = int(total * self.split_val_value)

            list_train = img_path[0:len_train]
            self.list_all_train_imgs += list_train
            list_label_train = [idx_class for _ in range(len_train)]
            self.list_all_train_labels += list_label_train

            list_test  = img_path[len_train: len_train + len_test]
            self.list_all_test_imgs += list_test
            list_label_test = [idx_class for _ in range(len_test)]
            self.list_all_test_labels += list_label_test

            list_val   = img_path[len_train + len_test: len_train + len_test + len_val]
            self.list_all_val_imgs += list_val
            list_label_val = [idx_class for _ in range(len_val)]
            self.list_all_val_labels += list_label_val
            
    def get_tranform(self, activate_augmentation = False):
        transform_list = [
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

       
        if activate_augmentation:
            transform_list.insert(1, transforms.RandomHorizontalFlip())  
            transform_list.insert(2, transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2))  

        transform = transforms.Compose(transform_list)

        return transform


###########################################################################################################################################################
if __name__ == "__main__":
    main_path = 'C:\\Users\\Alan\\Desktop\\projeto_neu_surface\\dataset_neu\\'
    load_data = LoadData(main_path)
    load_data.run()

