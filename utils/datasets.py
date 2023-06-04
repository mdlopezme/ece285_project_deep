from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import rawpy
import numpy as np

class LabeledDataset(Dataset):
    def __init__(self, root_dir, *csv_files, transform=None):
        """
        Point to the root directory of the dataset and the csv
        files containing the list of images and their corresponding labels.

        Input:
        root_dir: Root directory of the dataset
        csv_files: List of csv files containing the list of images and their corresponding labels
        transform: Optional transform to be applied on a sample

        The csv files should be in the following format:
        short_exposure_image long_exposure_image iso fstop

        Example:
        ./Sony/short/10003_00_0.04s.ARW ./Sony/long/10003_00_10s.ARW ISO200 F9
        ./Sony/short/10003_00_0.1s.ARW ./Sony/long/10003_00_10s.ARW ISO200 F9
        ./Sony/short/10003_01_0.04s.ARW ./Sony/long/10003_00_10s.ARW ISO200 F9
        ./Sony/short/10003_01_0.1s.ARW ./Sony/long/10003_00_10s.ARW ISO200 F9

        Output:
        image_short: Short exposure image
        image_long: Long exposure image
        label: Folder name of the image
        iso: ISO value of the image
        fstop: F-stop value of the image
        """
        self.df = pd.DataFrame()
        for csv_file in csv_files:
            if csv_file is None:
                continue
            self.df = pd.concat([self.df, pd.read_csv(csv_file, sep=' ', header=None)], ignore_index=True)
        self.root_dir = root_dir
        self.transform = transform

        self.short_exposure = self.df.iloc[:, 0]
        self.long_exposure = self.df.iloc[:, 1]
        self.iso = self.df.iloc[:, 2]
        self.fstop = self.df.iloc[:, 3]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        short_exposure = self.short_exposure[index]
        long_exposure = self.long_exposure[index]
        iso = self.iso[index]
        fstop = self.fstop[index]

        path_to_image_short = os.path.join(self.root_dir, short_exposure)
        path_to_image_long = os.path.join(self.root_dir, long_exposure)
        
        with rawpy.imread(path_to_image_short) as raw:
            image_short = raw.raw_image_visible.astype(np.float32)

        with rawpy.imread(path_to_image_long) as raw:
            image_long = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16).astype(np.float32)

        # Ratio of image lenght to width, before transform
        ratio = image_short.shape[1] / image_short.shape[0]
        
        if self.transform is not None:
            # BUG: If transform includes random crop, then the images will not be aligned
            image_short = self.transform(image_short)
            image_long = self.transform(image_long)

        # Extract folder name from path
        label = os.path.dirname(short_exposure).split('/')[1]
        

        return image_short, image_long, ratio, label, iso, fstop
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),
    ])
    csv_files = [
        'dataset/Sony_train_list.txt',
        'dataset/Fuji_train_list.txt'
        ]
    dataset = LabeledDataset('dataset',*csv_files, transform=transform)

    data = dataset[0]
    image = data[0]
    image2 = data[1]
    print(f'type: {type(image)}, shape: {image.shape}')   
    print(data[2])
    print(data[3])
    print(data[4])
    print(data[5])

    # Show images
    plt.figure()
    plt.imshow(image)
    plt.figure()
    plt.imshow(image2)
    plt.show()