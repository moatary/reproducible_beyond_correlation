
"""
Data download and preprocessing script.

Usage
-----
python utils/download_data.py

Downloads MNIST automatically.  For the Gender Face dataset,
place the raw images in data/raw/gender/ (one subfolder per class:
data/raw/gender/male/ and data/raw/gender/female/) then re-run
this script to produce data/imagestargets.pkl.
"""
import pathlib, pickle
import numpy as np
from torchvision import datasets, transforms

DATA_DIR = pathlib.Path('data')
DATA_DIR.mkdir(exist_ok=True)

print('Downloading MNIST...')
tf = transforms.Compose([transforms.ToTensor()])
datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tf)
datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
print('MNIST ready.')

gender_raw = DATA_DIR / 'raw' / 'gender'
if gender_raw.exists():
    import cv2
    from skimage.transform import resize as sk_resize
    TARGET_SIZE = (28, 28)
    images, labels = [], []
    for label_idx, cls in enumerate(['female', 'male']):
        cls_dir = gender_raw / cls
        for fpath in sorted(cls_dir.glob('*.jpg')) + sorted(cls_dir.glob('*.png')):
            img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = sk_resize(img, TARGET_SIZE, anti_aliasing=True)
            images.append(img.ravel().astype(np.float32))
            labels.append(label_idx)
    images = np.stack(images)
    labels = np.array(labels, dtype=np.int64)
    out = DATA_DIR / 'imagestargets.pkl'
    pickle.dump([images, labels], open(out, 'wb'))
    print(f'Gender Face dataset saved to {out} ({len(images)} samples).')
else:
    print('Gender Face raw images not found — skipping. See docstring for instructions.')
