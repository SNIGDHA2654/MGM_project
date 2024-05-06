"""
Utility functions for working with an Inception v3 model, including functions to calculate Inception Score and FID.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.inception import inception_v3
from scipy.linalg import sqrtm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
#import inception_utils
#from dataloader_1.dataset import CelebAMaskDataset
from torch.utils.data import ConcatDataset
import inception_utils 
import psutil

class CelebAMaskDataset(Dataset):
    def __init__(self, dataroot, is_label=True, phase='train'):
        """
        Initializes the CelebAMaskDataset.

        Parameters:
            dataroot (str): The root directory where the data is stored.
            is_label (bool): Whether the dataset should load labeled data (True) or unlabeled (False).
            phase (str): The phase of the dataset to use ('train', 'val', 'test', or 'train-val').
        """
        self.dataroot = dataroot
        self.is_label = is_label
        self.phase = phase
        self.idx_list = self._load_index_list()

    def __len__(self):
        return len(self.idx_list)
    
    def _load_index_list(self):
        """Load index list based on the phase, label requirement, and data root."""
        # Define the subdirectory based on whether data is labeled or not
#         data_root = os.path.join(self.dataroot, 'label_data' if self.is_label else 'unlabel_data')
        data_root = self.dataroot
        
        # Set the filename based on the dataset phase or the type of data
        if self.is_label:
            if self.phase == 'train':
                filename = 'train_full_list.txt'
            elif self.phase == 'val':
                filename = 'val_full_list.txt'
            elif self.phase == 'test':
                filename = 'test_list.txt'
            elif self.phase == 'train-val':
                self.tmp_data_path = data_root+ '/label_data'
                # Combine train and validation lists for labeled data
                train_list = self._load_file(self.tmp_data_path+'/train_full_list.txt')
                val_list = self._load_file(self.tmp_data_path+'/val_full_list.txt')
                return np.concatenate((train_list, val_list))
            else:
                raise ValueError("Invalid phase specified for labeled data.")
    
            # Handle the unlabeled data
            #filename = 'unlabel_list.txt'  # Default for all phases if unlabeled
        
        self.tmp_data_path = data_root+ '/unlabel_data'
        file_path = self.tmp_data_path + "/unlabel_list.txt"
        return self._load_file(file_path)

    def _load_file(self, file_path):
        """Helper function to load a file and raise an error if not found."""
#         print(file_path)
#         print(os.path.exists(file_path))
#         print(type(file_path))
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Expected file at {file_path} not found.")
        return np.loadtxt(file_path, dtype=str)

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        """Retrieve an item by its index."""
        img_path = self.tmp_data_path +'/images/' + self.idx_list[idx]
#         print("THE PATH OF IMAGE IS", img_path)
        img = Image.open(img_path).convert('RGB')
        return {'image': np.array(img)}

"""
This module computes Inception scores and other metrics for datasets using an Inception v3 model.
"""
def check_resource_usage():
    """Check system resource usage."""
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    disk_percent = psutil.disk_usage('/').percent
    
    print(f"CPU Usage: {cpu_percent}%")
    print(f"Memory Usage: {memory_percent}%")
    print(f"Disk Usage: {disk_percent}%")
    
    if cpu_percent > 90:
        print("Warning: High CPU usage detected.")
    if memory_percent > 90:
        print("Warning: High memory usage detected.")
    if disk_percent > 90:
        print("Warning: High disk usage detected.")

@torch.no_grad()

def extract_features(loader, inception, device):
    print('inside Extract')
    """Extract features from the data loader using the inception model."""
    pools, logits = [], []
    try:
        print("extractor")
        for data in loader:
            print("h")
            img = data['image']
            img = img.permute(0, 3, 1, 2).float()
            img = img.to(device)
            if img.shape[1] != 3:
                img = img.expand(-1, 3, -1, -1)  # Ensure 3 color channels
            print("Processing image...")
            pool_val, logits_val = inception(img)
            pools.append(pool_val.cpu().numpy())
            logits.append(torch.nn.functional.softmax(logits_val, dim=1).detach().cpu().numpy())

        pools = np.concatenate(pools, axis=0)
        logits = np.concatenate(logits, axis=0)
    except MemoryError:
        print("MemoryError: The script was killed due to insufficient memory in the extract method.")
        sys.exit(1)
    except RuntimeError as e:
        print(f"RuntimeError: The script was killed in the extract method - {e}")
        sys.exit(1)
    print("Outside extract")
    return pools, logits


def get_dataset(path, dataset_name):
    """Load the labeled and unlabeled dataset based on the input arguments."""
    if dataset_name == 'celeba-mask':
        # Assuming that the labeled and unlabeled data are stored in the same root path but in different subdirectories
        try:
            labeled_dataset = CelebAMaskDataset( path, is_label=True, phase='train-val')
            unlabeled_dataset = CelebAMaskDataset( path, is_label=False)

    #         # Optionally, you can load only labeled or only unlabeled depending on another argument or condition
    #         if args.use_both_types:
                # Concatenate both datasets into a single dataset
            dataset = ConcatDataset([labeled_dataset, unlabeled_dataset])
            print("NO errors")
        except Exception as e:
            print("get_dataset error:", e)
        
#         else:
#             # Decide based on some condition or configuration
#             dataset = labeled_dataset if args.prefer_labeled else unlabeled_dataset
    else:
        raise Exception('No such dataset loader defined.')
    
    return dataset

def main():
    print("Inside main")
    """Main function to compute Inception features and scores."""
    try:
        # Configuration parameters
        size = 256
        batch = 2
        #n_sample = 50000
        output = 'output.pkl'
        image_mode = 'RGB'
        dataset_name = 'celeba-mask'
        path = '/home2/megha/unlabel_data'

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inception = inception_utils.load_inception_net()
        inception = inception.to(device)
        print(type(inception))
        
        dataset = get_dataset(path, dataset_name)
        print("Length of the dataset:", len(dataset))
        dataset = torch.utils.data.Subset(dataset, range(8))
        print("Length of the dataset:", len(dataset))
        #rint("Dataset Class Initiated")
        loader = DataLoader(dataset, batch_size=batch, num_workers=0)
        for data in loader:
            print("h")
        
        #rint("Dataloader Created ...")
    #     logits = extract_features(loader, inception, device)
    #     print(f'Extracted {logits.shape[0]} features')

    #     IS_mean, IS_std = calculate_inception_score(logits)
    #     print(f'Training data from dataloader has an Inception Score of {IS_mean:.5f} +/- {IS_std:.5f}')
    #     print('Calculating means and covariances...')
    #     mean, cov = np.mean(logits, axis=0), np.cov(logits, rowvar=False)

    #     with open(output, 'wb') as f:
    #         pickle.dump({'mean': mean, 'cov': cov, 'size': size, 'path': path}, f)
            
        try:
            check_resource_usage()  # Check resource usage before extraction
            pools, logits = extract_features(loader, inception, device)
            check_resource_usage()  # Check resource usage after extraction
        except Exception as e:
            print("An error occurred during feature extraction:", e)
        print(f'Extracted {pools.shape[0]} features')
        IS_mean, IS_std = inception_utils.calculate_inception_score(logits)
        print(f'Training data from dataloader has an Inception Score of {IS_mean:.5f} +/- {IS_std:.5f}')
        print('Calculating means and covariances...')
        mean, cov = np.mean(pools, axis=0), np.cov(pools, rowvar=False)
        output_filename = "/home2/megha/semanticGAN_code/semanticGAN/inception_output.pkl"

        # Manually define other parameters
        resoln_size = 1024
        dataset_path = "/home2/megha/unlabel_data"

        # Use the defined values to save the output
        with open(output_filename, 'wb') as f:
            pickle.dump({'mean': mean, 'cov': cov, 'size': resoln_size, 'path': dataset_path}, f)

        print("Pickle file created successfully.")

        # Open and print contents of the pickle file
        with open(output_filename, 'rb') as f:
            data = pickle.load(f)
            print("Contents of the pickle file:")
            print(data)
    except MemoryError:
        print("MemoryError: The script was killed due to insufficient memory.")
        sys.exit(1)
    except RuntimeError as e:
        print(f"RuntimeError: The script was killed - {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
