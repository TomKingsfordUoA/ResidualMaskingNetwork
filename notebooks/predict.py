import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from main_fer2013 import get_model, get_dataset
from trainers.tta_trainer import FER2013Trainer

# Correct the cwd:
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

config_path = 'configs/fer2013_config.json'
class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

df_train = pd.read_csv('data/train.csv')
df_val = pd.read_csv('data/val.csv')
df_test = pd.read_csv('data/test.csv')

print(df_train.shape)
print(df_val.shape)
print(df_test.shape)

configs = json.load(open(config_path))
configs['device'] = 'cuda:0'
configs['cwd'] = os.getcwd()

device = torch.device(configs['device'])

train_set, val_set, test_set = get_dataset(configs)
model_arch = get_model(configs=configs)
trainer = FER2013Trainer(
    model=model_arch,
    train_set=train_set,
    val_set=val_set,
    test_set=test_set,
    configs=configs,
)

state = torch.load('checkpoint/resnet34_test_2021Nov16_13.37', map_location=device)
trainer._model.load_state_dict(state["net"])
trainer._model.eval()


def predict_dataset(dataset_loader, tta: bool, verbose=0):
    with torch.no_grad():
        targets_batches = []
        outputs_batches = []
        for i, (images, targets) in tqdm(
            enumerate(dataset_loader), total=len(dataset_loader), leave=False
        ):
            if tta:
                images = torch.cat(images).to(device=device)
                targets = targets.repeat(len(images)).to(device=device)
            else:
                images = images.to(device=device)
                targets = targets.to(device=device)

            outputs = trainer._model(images)

            if verbose:
                print(images.shape)
                print(targets.shape)
                print(outputs.shape)

            targets_batches.append(targets.cpu().numpy())
            outputs_batches.append(outputs.cpu().numpy())

    targets = np.concatenate(targets_batches)
    outputs = np.concatenate(outputs_batches)

    if verbose:
        print(targets.shape)
        print(outputs.shape)

    return targets, outputs


train_y, train_y_pred = predict_dataset(trainer._train_loader, tta=False)
test_y, test_y_pred = predict_dataset(trainer._test_loader, tta=True)

df_train = pd.DataFrame(data=train_y_pred, columns=class_names)
df_train['gt'] = train_y
df_train['gt'] = df_train['gt'].apply(lambda gt_index: class_names[gt_index])

df_test = pd.DataFrame(data=test_y_pred, columns=class_names)
df_test['gt'] = test_y
df_test['gt'] = df_test['gt'].apply(lambda gt_index: class_names[gt_index])

print(df_train.shape)
print(df_test.shape)
print(df_train.head())
print(df_test.head())

df_test.to_csv('data/test_pred.csv', index=True, header=True)
df_train.to_csv('data/train_pred.csv', index=True, header=True)
