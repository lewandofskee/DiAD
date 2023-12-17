import csv
import json
import cv2
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]
class VisaDataset(Dataset):
    def __init__(self,type, root):
        self.data = []
        with open('./training/VisA/visa.csv', 'rt') as f:
            render = csv.reader(f, delimiter=',')
            header = next(render)
            for row in render:
                if row[1] == type:
                    data_dict = {'object':row[0],'split':row[1],'label':row[2],'image':row[3],'mask':row[4]}
                    self.data.append(data_dict)
        self.label_to_idx = {'candle': '0', 'capsules': '1', 'cashew': '2', 'chewinggum': '3', 'fryum': '4', 'macaroni1': '5',
                             'macaroni2': '6', 'pcb1': '7', 'pcb2': '8', 'pcb3': '9', 'pcb4': '10',
                             'pipe_fryum': '11',}
        self.image_size = (256,256)
        self.root = root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['image']
        target_filename = item['image']
        prompt = ""
        if item.get("mask", None):
            mask = cv2.imread( self.root + item['mask'], cv2.IMREAD_GRAYSCALE)
        else:
            if item['label'] == 'normal':  # good
                mask = np.zeros(self.image_size).astype(np.uint8)
            elif item['label'] == 'anomaly':  # defective
                mask = (np.ones(self.image_size)).astype(np.uint8)
            else:
                raise ValueError("Labels must be [None, 0, 1]!")

        source = cv2.imread(self.root + source_filename)
        target = cv2.imread(self.root + target_filename)
        source = cv2.cvtColor(source, 4)
        target = cv2.cvtColor(target, 4)
        source = Image.fromarray(source, "RGB")
        target = Image.fromarray(target, "RGB")
        mask = Image.fromarray(mask, "L")
        transform_fn = transforms.Resize(self.image_size)
        source = transform_fn(source)
        target = transform_fn(target)
        mask = transform_fn(mask)
        source = transforms.ToTensor()(source)
        target = transforms.ToTensor()(target)
        mask = transforms.ToTensor()(mask)
        normalize_fn = transforms.Normalize(mean=mean_train, std=std_train)
        source = normalize_fn(source)
        target = normalize_fn(target)
        clsname = item["object"]
        image_idx = self.label_to_idx[clsname]

        return dict(jpg=target, txt=prompt, hint=source, mask=mask, filename=source_filename, clsname=clsname, label=int(image_idx))

