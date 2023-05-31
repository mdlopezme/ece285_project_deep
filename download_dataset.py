import os
import urllib.request
import zipfile
import progressbar

pbar = None
def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None

# Download dataset from https://storage.googleapis.com/isl-datasets/SID/Sony.zip
# and https://storage.googleapis.com/isl-datasets/SID/Fuji.zip
os.makedirs('dataset', exist_ok=True)
# Show progress bar
print("Downloading Sony Dataset")
urllib.request.urlretrieve('https://storage.googleapis.com/isl-datasets/SID/Sony.zip', 'dataset/Sony.zip', show_progress)

print("Downloading Fuji Dataset")
urllib.request.urlretrieve('https://storage.googleapis.com/isl-datasets/SID/Fuji.zip', 'dataset/Fuji.zip', show_progress)


# Unzip the downloaded files
with zipfile.ZipFile('dataset/Sony.zip', 'r') as zip_ref:
    zip_ref.extractall('dataset')
with zipfile.ZipFile('dataset/Fuji.zip', 'r') as zip_ref:
    zip_ref.extractall('dataset')

# Remove the zip files
os.remove('dataset/Sony.zip')
os.remove('dataset/Fuji.zip')

