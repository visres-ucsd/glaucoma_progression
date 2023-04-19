import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
from dataset import ImageDataset, ImageRNFLDataset, ImageRNFLSeparateDataset, ImageRNFLVFDataset
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from vit import ViT
from vit_mmodal import ViT_rnfl, ViT_rnfl_vf
from recorder import Recorder
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from datetime import datetime

# Define the normalization parameters
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
# Define the inverse normalization transform
inv_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(mean, std)],
    std=[1/s for s in std]
)

model_path = "________.pth"

df_oct = pd.read_csv("_______.csv", index_col=0)
df_oct_test = df_oct[df_oct["cohort"] == "test"]
test_paths = ["/projects/ps-visres-group/raw-bscans/{}".format(path) for path in df_oct_test["path"]]

# rnfl thickness for multimodal
test_rnfl_paths = ["../metadata/rnfl_thickness/{}.npy".format(path.split("/")[1][:-4]) for path in df_oct_test["path"]]

# vf for multimodal
test_vf_paths = ["../metadata/visual_fields/{}.npy".format(row["IDEYE"] + "_" + row["date_first_progression"]) for _,row in df_oct_test.iterrows()]

label_map = {"Healthy":0.0, "GVFD & GON":1.0, "GVFD":1.0}
test_labels = [label_map[x] for x in df_oct_test["glaucoma"]]

# if not multimodal
# test_dataset = ImageDataset(test_paths, test_labels, data_transforms)

# if multimodal
# test_dataset = ImageRNFLDataset(test_paths, test_rnfl_paths, test_labels, data_transforms)

# if multimodal_custom
# test_dataset = ImageRNFLSeparateDataset(test_paths, test_rnfl_paths, test_labels, data_transforms)

# if multimodal rnfl and vf
test_dataset = ImageRNFLVFDataset(test_paths, test_rnfl_paths, test_vf_paths, test_labels, data_transforms)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1) # shuffle false!

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = ViT(
#     image_size = 224,
#     patch_size = 16,
#     num_classes = 1,
#     dim = 768,
#     depth = 12,
#     heads = 12,
#     mlp_dim = 3072,
#     dropout = 0.0,
#     emb_dropout = 0.0,
# )
# model.mlp_head.append(torch.nn.Sigmoid())

# model = ViT_rnfl(
#     image_size = 224,
#     patch_size = 16,
#     num_classes = 1,
#     dim = 768,
#     depth = 12,
#     heads = 12,
#     mlp_dim = 3072,
#     dropout = 0.0,
#     emb_dropout = 0.0,
#     rnfl_len = 768,
#     num_segments = 128,
# )
# model.mlp_head.append(torch.nn.Sigmoid())

# custom mmodal vit with rnfl and vf
model = ViT_rnfl_vf(
    image_size = 224,
    patch_size = 16,
    num_classes = 1,
    dim = 768,
    depth = 12,
    heads = 12,
    mlp_dim = 3072,
    dropout = 0.0,
    emb_dropout = 0.0,
    rnfl_len = 768,
    rnfl_num_segments = 128,
    vf_len = 156,
    vf_num_segments = 39,
)
model.mlp_head.append(torch.nn.Sigmoid())


# model = models.vit_b_16(image_size=224)
# num_ftrs = model.heads.head.in_features
# model.heads = torch.nn.Sequential(
#     torch.nn.Linear(in_features=num_ftrs,out_features=1),   
#     torch.nn.Sigmoid()
# )
# model = models.resnet50()
# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Sequential(
#     torch.nn.Linear(in_features=num_ftrs,out_features=1),   
#     torch.nn.Sigmoid()
# )
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

threshold = 0.5
running_corrects = 0
all_preds_logits = []
all_preds = []
all_labels = []
# Iterate over data.
for inputs, labels in test_loader:
    if type(inputs) == list:
        inputs_list = []
        for x in inputs:
            inputs_list.append(x.to(device))
        inputs = inputs_list
    else:    
        inputs = inputs.to(device)
    labels = labels.type(torch.LongTensor)
    labels = labels.to(device)

    outputs = model(inputs)
    preds = torch.where(outputs > threshold, 1., 0.)

    running_corrects += torch.sum(preds == labels.data)

    all_preds.append(preds.item())
    all_preds_logits.append(outputs.item())
    all_labels.append(labels.item())

test_acc = running_corrects.double() / len(test_dataset)
print("test accuracy: ", test_acc.item())
auroc = roc_auc_score(all_labels, all_preds_logits)
print("auroc: ", auroc)

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

# Find optimal probability threshold
threshold = Find_Optimal_Cutoff(all_labels, all_preds_logits)[0]
# print threshold
# [0.31762762459360921]

# Find prediction to the dataframe applying threshold
all_preds = list(map(lambda x: 1 if x > threshold else 0, all_preds_logits))

tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
sensitivity = tp / (tp+fn)
specificity = tn / (tn+fp)
print("sensitivity: {}, specificity: {}".format(sensitivity, specificity))

np.save("{}_preds".format(model_path[:-4]), {"labels":all_labels, "all_preds_logits":all_preds_logits})