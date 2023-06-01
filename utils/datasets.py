from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import rawpy

class SonyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file, header=None, sep=' ')
        self.root_dir = root_dir
        self.transform = transform

        self.short_exposure = self.df.iloc[:, 0]
        self.long_exposure = self.df.iloc[:, 1]
        # self.iso = self.df.iloc[:, 2]
        # self.fstop = self.df.iloc[:, 3]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        short_exposure = self.short_exposure[index]
        long_exposure = self.long_exposure[index]
        # iso = self.iso[index]
        # fstop = self.fstop[index]

        path_to_image = os.path.join(self.root_dir, short_exposure)
        
        with rawpy.imread(path_to_image) as raw:
            image = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        
        # if self.transform is not None:
        #     image = self.transform(image)

        return image, short_exposure