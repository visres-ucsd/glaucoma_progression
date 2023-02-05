import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from torch.utils.data import Dataset, DataLoader
from dataset import ImageDataset, ImageRNFLDataset, ImageRNFLVFDataset, ImageRNFLVFLongitudinalDataset
import torchvision
from torchvision import models, transforms
import numpy as np
import pandas as pd
import time
import copy
import tqdm 
from tqdm import tqdm
from vit import ViT
from vit_mmodal import ViT_rnfl, ViT_rnfl_vf, ViT_rnfl_vf_longitudinal

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((530,530)),
        transforms.RandomCrop((512,512)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_test': transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

batch_size = 16
threshold = 0.5 # for inference
save_path = "model_checkpoint_vit"

df_oct = pd.read_csv("metadata_file.csv", index_col=0)

df_oct_train = df_oct[df_oct["Cohort"] == "Train"]
df_oct_val = df_oct[df_oct["Cohort"] == "Validation"]
df_oct_test = df_oct[df_oct["Cohort"] == "Test"]

train_paths = ["/projects/ps-visres-group/raw-bscans/{}".format(path) for path in df_oct_train["path"]]
val_paths = ["/projects/ps-visres-group/raw-bscans/{}".format(path) for path in df_oct_val["path"]]
test_paths = ["/projects/ps-visres-group/raw-bscans/{}".format(path) for path in df_oct_test["path"]]

# rnfl thickness for multimodal
train_rnfl_paths = ["../metadata/rnfl_thickness/{}.npy".format(path.split("/")[1][:-4]) for path in df_oct_train["path"]]
val_rnfl_paths = ["../metadata/rnfl_thickness/{}.npy".format(path.split("/")[1][:-4]) for path in df_oct_val["path"]]
test_rnfl_paths = ["../metadata/rnfl_thickness/{}.npy".format(path.split("/")[1][:-4]) for path in df_oct_test["path"]]

# vf for multimodal
train_vf_paths = ["../metadata/visual_fields/{}.npy".format(row["IDEYE"] + "_" + row["date_first_progression"]) for i,row in df_oct_train.iterrows()]
val_vf_paths = ["../metadata/visual_fields/{}.npy".format(row["IDEYE"] + "_" + row["date_first_progression"]) for i,row in df_oct_val.iterrows()]
test_vf_paths = ["../metadata/visual_fields/{}.npy".format(row["IDEYE"] + "_" + row["date_first_progression"]) for i,row in df_oct_test.iterrows()]

### for time series------------------------------------------------------------------------------------------------------
train_paths2 = ["/projects/ps-visres-group/raw-bscans/{}".format(path) for path in df_oct_train["path2"]]
val_paths2 = ["/projects/ps-visres-group/raw-bscans/{}".format(path) for path in df_oct_val["path2"]]
test_paths2 = ["/projects/ps-visres-group/raw-bscans/{}".format(path) for path in df_oct_test["path2"]]

# rnfl thickness for multimodal
train_rnfl_paths2 = ["../metadata/rnfl_thickness/{}.npy".format(path.split("/")[1][:-4]) for path in df_oct_train["path2"]]
val_rnfl_paths2 = ["../metadata/rnfl_thickness/{}.npy".format(path.split("/")[1][:-4]) for path in df_oct_val["path2"]]
test_rnfl_paths2 = ["../metadata/rnfl_thickness/{}.npy".format(path.split("/")[1][:-4]) for path in df_oct_test["path2"]]

# vf for multimodal
train_vf_paths2 = ["../metadata/visual_fields/{}.npy".format(row["IDEYE"] + "_" + row["date_second_progression"]) for i,row in df_oct_train.iterrows()]
val_vf_paths2 = ["../metadata/visual_fields/{}.npy".format(row["IDEYE"] + "_" + row["date_second_progression"]) for i,row in df_oct_val.iterrows()]
test_vf_paths2 = ["../metadata/visual_fields/{}.npy".format(row["IDEYE"] + "_" + row["date_second_progression"]) for i,row in df_oct_test.iterrows()]

train_time_deltas = [t for t in df_oct_train["time_delta"]]
val_time_deltas = [t for t in df_oct_val["time_delta"]]
test_time_deltas = [t for t in df_oct_test["time_delta"]]
### ----------------------------------------------------------------------------------------------------------------------


label_map = {"stable":0.0, "progressing":1.0}
train_labels = [label_map[x] for x in df_oct_train["patient_level_glaucoma"]]
val_labels = [label_map[x] for x in df_oct_val["patient_level_glaucoma"]]
test_labels = [label_map[x] for x in df_oct_test["patient_level_glaucoma"]]

## if not multimodal
# train_dataset = ImageDataset(train_paths, train_labels, data_transforms['train'])
# val_dataset = ImageDataset(val_paths, val_labels, data_transforms['val_test'])
# test_dataset = ImageDataset(test_paths, test_labels, data_transforms['val_test'])

## if multimodal
# train_dataset = ImageRNFLDataset(train_paths, train_rnfl_paths, train_labels, data_transforms['train'])
# val_dataset = ImageRNFLDataset(val_paths, val_rnfl_paths, val_labels, data_transforms['val_test'])
# test_dataset = ImageRNFLDataset(test_paths, test_rnfl_paths, test_labels, data_transforms['val_test'])

# # if multimodal rnfl and vf
# train_dataset = ImageRNFLVFDataset(train_paths, train_rnfl_paths, train_vf_paths, train_labels, data_transforms['train'])
# val_dataset = ImageRNFLVFDataset(val_paths, val_rnfl_paths, val_vf_paths, val_labels, data_transforms['val_test'])
# test_dataset = ImageRNFLVFDataset(test_paths, test_rnfl_paths, test_vf_paths, test_labels, data_transforms['val_test'])

# if multimodal rnfl and vf and longitudinal 2 timepoints
train_dataset = ImageRNFLVFLongitudinalDataset(train_paths, train_paths2, train_rnfl_paths, train_rnfl_paths2, train_vf_paths, train_vf_paths2, train_time_deltas, train_labels, data_transforms['train'])
val_dataset = ImageRNFLVFLongitudinalDataset(val_paths, val_paths2, val_rnfl_paths, val_rnfl_paths2, val_vf_paths, val_vf_paths2, val_time_deltas, val_labels, data_transforms['val_test'])
test_dataset = ImageRNFLVFLongitudinalDataset(test_paths, test_paths2, test_rnfl_paths, test_rnfl_paths2, test_vf_paths, test_vf_paths2, test_time_deltas, test_labels, data_transforms['val_test'])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
dataloaders = {"train":train_loader, "val":val_loader, "test":test_loader}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_sizes = {"train":len(train_dataset), "val":len(val_dataset), "test":len(test_dataset)}
class_names = ["stable", "progressing"]

print('The shape of tensor for 50th image in train dataset: ',train_dataset[49][0][0].shape)
print('The label for 50th image in train dataset: ',train_dataset[49][1])
print('Number of train, val, test images: ', len(train_dataset), len(val_dataset), len(test_dataset))

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 100.

    losses = {"train":[], "val":[], "test":[]}
    accs = {"train":[], "val":[], "test":[]}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_count = 0

            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                # Iterate over data.
                # for inputs, labels in dataloaders[phase]:
                for inputs, labels in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    if type(inputs) == list:
                        inputs_list = []
                        for x in inputs:
                            inputs_list.append(x.to(device))
                        inputs = inputs_list
                    else:
                        inputs = inputs.to(device)
                    labels = labels.float()
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # outputs = torch.squeeze(model(inputs))
                        outputs = torch.reshape(model(inputs), labels.shape)
                        preds = torch.where(outputs > threshold, 1., 0.)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    if type(inputs) == list:
                        running_loss += loss.item() * inputs[0].size(0)
                    else:
                        running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    running_count += batch_size
                    running_acc = running_corrects/running_count * 100.

                    tepoch.set_postfix(loss=loss.item(), accuracy=running_acc.item())
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            losses[phase].append(epoch_loss)
            accs[phase].append(epoch_acc.cpu().detach().item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), f'{save_path}.pth')
        
        np.save(f'{save_path}_loss.npy', losses)
        np.save(f'{save_path}_accs.npy', accs)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best test Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# custom vit
model_ft = ViT(
    image_size = 224,
    patch_size = 16,
    num_classes = 1,
    dim = 768,
    depth = 12,
    heads = 12,
    mlp_dim = 3072,
    dropout = 0.0,
    emb_dropout = 0.0
)
model_ft.mlp_head.append(torch.nn.Sigmoid())
print(model_ft)

# custom mmodal vit
# model_ft = ViT_rnfl(
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
# model_ft.mlp_head.append(torch.nn.Sigmoid())
# print(model_ft)

# # custom mmodal vit with rnfl and vf
# model_ft = ViT_rnfl_vf(
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
#     rnfl_num_segments = 128,
#     vf_len = 156,
#     vf_num_segments = 39,
# )
# model_ft.mlp_head.append(torch.nn.Sigmoid())

# custom mmodal vit with rnfl and vf and 2 timepoints
# model_ft = ViT_rnfl_vf_longitudinal(
#     image_size = 512,
#     patch_size = 32,
#     num_classes = 1,
#     dim = 768,
#     depth = 12,
#     heads = 12,
#     mlp_dim = 3072,
#     dropout = 0.0,
#     emb_dropout = 0.0,
#     rnfl_len = 768,
#     rnfl_num_segments = 128,
#     vf_len = 156,
#     vf_num_segments = 156,
# )
# model_ft.mlp_head.append(torch.nn.Sigmoid())

# model_ft = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT, image_size=512)
# model_ft = models.vit_b_16(weights=None, image_size=240)
# model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# model_ft = models.resnet50(weights=None)
# print(model_ft)
# num_ftrs = model_ft.heads.head.in_features
# model_ft.heads = torch.nn.Sequential(
#     torch.nn.Linear(in_features=num_ftrs,out_features=1),   
#     torch.nn.Sigmoid()
# )

# model_ft.load_state_dict(torch.load(pretrained_model_path)) # load pretrain weights

# for param in model_ft.parameters(): # freeze
#     param.requires_grad = False
# for param in model_ft.mlp_head.parameters(): # freeze
#     param.requires_grad = True 

# model_ft.fc = torch.nn.Sequential(
#     torch.nn.Linear(in_features=num_ftrs,out_features=1),   
#     torch.nn.Sigmoid()
# )

# model_ft = models.resnet50(weights=None)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = torch.nn.Sequential(
#     torch.nn.Linear(in_features=num_ftrs,out_features=1)
# )
# model_ft.load_state_dict(torch.load(pretrained_model_path)) # load pretrain weights
# # for param in model_ft.parameters(): # freeze
#     # param.requires_grad = False
# model_ft.fc = torch.nn.Sequential(
#     torch.nn.Linear(in_features=num_ftrs,out_features=1),   
#     torch.nn.Sigmoid()
# )
model_ft = model_ft.to(device)
criterion = nn.BCELoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.00005)

# Decay LR by a factor of 0.5 every 20 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=1.)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=100)