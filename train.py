from share import *
import torch
import random
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from mvtecad_dataloader import MVTecDataset
from sgn.logger import ImageLogger
from sgn.model import create_model, load_state_dict
from visa_dataloader import VisaDataset
from pytorch_lightning.callbacks import ModelCheckpoint
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministirc = True
    torch.backends.cudnn.benchmark = False

# Configs
resume_path = './models/diad.ckpt'

setup_seed(1)
batch_size = 12
logger_freq = 3000000000000
learning_rate = 1e-5
only_mid_control = True
data_path = '/root/autodl-tmp/mvtecad/'

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('models/diad.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'),strict=False)
model.learning_rate = learning_rate
model.only_mid_control = only_mid_control

# Misc
train_dataset, test_dataset = MVTecDataset('train',data_path), MVTecDataset('test',data_path)
# train_dataset, test_dataset = VisaDataset('train',data_path), VisaDataset('test',data_path)
train_dataloader = DataLoader(train_dataset, num_workers=8, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, num_workers=8, batch_size=1, shuffle=True)

ckpt_callback_val_loss = ModelCheckpoint(monitor='val_acc', dirpath='./val_ckpt/',mode='max')
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger,ckpt_callback_val_loss], accumulate_grad_batches=4, check_val_every_n_epoch=25)

# Train!
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)