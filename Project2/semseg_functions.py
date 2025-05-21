import importlib
import seUNetTrans_Implementation
importlib.reload(seUNetTrans_Implementation)

# import cv2
# from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
# from scipy import ndimage
# from skimage import measure, color, io
import glob
import numpy as np
# import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp
# from PIL import Image
# import tqdm
# import pandas as pd
# import pytorch_lightning as pl
import os
import copy
from skimage.io import imread
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader
from seUNetTrans_Implementation import SEUNetTrans

batch_size = 10

# This function is preparing image data for training and validating machine learning models, particularly in the context of semantic segmentation tasks.
def load_imgs_labels(train_dir="./train",val_dir="./val"):
    # the glob module retrieve all PNG images in the specified directories train_dir/ val_dir.
    # The imread function read these images into numpy arrays. 
    # The stack function stacks these arrays into a single numpy array.
    # The single numpy array is then converted into a PyTorch tensor.
    train_imgs=np.stack(list(map(imread,sorted(glob.glob(os.path.join(train_dir,"imgs","*.png"))))))
    X_train=torch.FloatTensor(train_imgs).permute((0,3,1,2))/255

    # same as above, but for the validation image set
    val_imgs=np.stack(list(map(imread,sorted(glob.glob(os.path.join(val_dir,"imgs","*.png"))))))
    X_val=torch.FloatTensor(val_imgs).permute((0,3,1,2))/255
    
    # same as above, but for the training label set
    train_lbls=np.stack(list(map(lambda x: imread(x)[...,0].astype(int),sorted(glob.glob(os.path.join(train_dir,"labels","*.png"))))))
    Y_train=torch.LongTensor(train_lbls)
    
    # same as above, but for the validation label set
    val_lbls=np.stack(list(map(lambda x: imread(x)[...,0].astype(int),sorted(glob.glob(os.path.join(val_dir,"labels","*.png"))))))
    Y_val=torch.LongTensor(val_lbls)
    
    # return the training and validation image and label sets
    return X_train,Y_train,X_val,Y_val

# This function is used to train a machine learning model for semantic segmentation tasks.
# The function takes as input the training and validation image and label sets, the number of epochs, the model key, the encoder name, the path directory, and the device.
def train_model(X_train, Y_train, X_val, Y_val, save=True, n_epochs=10, model_key="unet", encoder_name="resnet18", path_dir = "./seg_models", device=None): 
    
    # Initialize the AMP Scaler
    # scaler = GradScaler()

    # set the device to GPU to speed up the training process if it is available, otherwise set it to CPU.
    if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data=TensorDataset(X_train,Y_train)
    val_data=TensorDataset(X_val,Y_val)

    # DataLoader is used to load the training and validation data in batches.
    train_dataloader=DataLoader(train_data,batch_size,shuffle=True)
    train_dataloader_ordered=DataLoader(train_data,batch_size,shuffle=False)
    val_dataloader=DataLoader(val_data,batch_size,shuffle=False)
    encoder_name="resnet18" if encoder_name not in smp.encoders.get_encoder_names() else encoder_name

    # model selection

    # The model is selected based on the model key. The model key is used to select the model from the segmentation_models_pytorch library.
    model_dict = {
        "unet": smp.Unet,
        "fpn": smp.FPN,
        "unetplusplus": smp.UnetPlusPlus,
        "seunettrans": SEUNetTrans
    }
    model = model_dict.get(model_key, smp.Unet)
    # The model is initialized with the number of classes and input channels, the encoder name, and the encoder weights.
    model = model(encoder_name=encoder_name, classes=3, in_channels=3).to(device)

    # The class weights are used to balance the class distribution in the training data.
    class_weight=compute_class_weight(class_weight='balanced', classes=np.unique(Y_train.numpy().flatten()), y=Y_train.numpy().flatten())
    class_weight=torch.FloatTensor(class_weight).to(device)

    # Training a machine learning model for semantic segmentation tasks
    # The loss function measures the difference between the predicted outputs of the model and the true labels. 
    # The goal is to minimize this loss.
    if model_key == "seunettrans":
        # The learning rate(lr) controls how fast the model learns — specifically, 
        # it determines the size of the steps the optimizer takes when updating weights to reduce loss.
        print("model= seunettrans. Using Combo Loss and Lower Learning Rate")
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)  
        loss_fn = ComboLoss(weight_dice=0.6)
    else:
        loss_fn=nn.CrossEntropyLoss(weight=class_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  
    if not os.path.exists(path_dir): os.makedirs(path_dir,exist_ok=True) 
    min_loss=np.inf
    best_model_epoch = None
    best_model_batch = None

    # Lists to store training and validation losses
    training_losses = []
    validation_losses = []

    for epoch in range(1,n_epochs+1):
        # training set 
        model.train(True)

        epoch_training_loss = []
        
        for i,(x,y_true) in enumerate(train_dataloader):
            x, y_true = x.to(device), y_true.to(device)
            optimizer.zero_grad()

            y_pred=model(x)
            loss=loss_fn(y_pred,y_true)

            loss.backward()
            optimizer.step()

            """ Add AMP
            with autocast():
                y_pred = model(x)
                loss = loss_fn(y_pred, y_true)

            # Backprop with gradient scaling
            scaler.scale(loss).backward()

            # Optimizer step via scaler
            scaler.step(optimizer)
            scaler.update()
            # end of AMP
            """
            epoch_training_loss.append(loss.item())
            print(f"Model: {model_key}, Encoder: {encoder_name}. Training: Epoch {epoch}, Batch {i}, Loss: {round(loss.item(),3)}")

        # Calculate average training loss for the epoch
        avg_training_loss = np.mean(epoch_training_loss)
        training_losses.append(avg_training_loss)

        # validation set
        model.train(False)
        with torch.no_grad():
            epoch_validation_loss = []
            for i,(x,y_true) in enumerate(val_dataloader):
                x, y_true = x.to(device), y_true.to(device)
                y_pred=model(x)
                loss=loss_fn(y_pred,y_true)
                epoch_validation_loss.append(loss.item())

            # Calculate average validation loss for the epoch
            avg_validation_loss = np.mean(epoch_validation_loss)
            validation_losses.append(avg_validation_loss)

            print(f"Val: Epoch {epoch}, Loss: {round(avg_validation_loss, 3)}")

            # Save the model with the lowest validation loss
            if avg_validation_loss < min_loss:
                min_loss = avg_validation_loss
                best_model = copy.deepcopy(model.state_dict())
                best_model_epoch = epoch
                best_model_batch = i
                if save:
                    with open(path_dir + f'/{epoch}_{i}_model.pkl', "w") as f:
                        torch.save(model.state_dict(), path_dir + f'/{epoch}_{i}_model.pkl')

    model.load_state_dict(best_model)
    print(f"Best model saved from Epoch {best_model_epoch}, Batch {best_model_batch}")
    return model, training_losses, validation_losses

# This function is used to make predictions on the validation image set using a trained machine learning model.
# The function takes as input the validation image set, the trained model, a boolean variable to save the model, the path directory, the model key, the encoder name, and the device.
def make_predictions(X_val,model=None,save=True,path_dir = "./seg_models", model_key="unet", encoder_name="resnet18", device=None):
    if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
    val_data=TensorDataset(X_val)
    val_dataloader=DataLoader(val_data,batch_size,shuffle=False)
    predictions=[]
    encoder_name="resnet18" if encoder_name not in smp.encoders.get_encoder_names() else encoder_name
    # load most recent saved model
    if model is None and save:
        model_dict = {
            "unet": smp.Unet,
            "fpn": smp.FPN,
            "unetplusplus": smp.UnetPlusPlus,
            "seunettrans": SEUNetTrans
        }
        model = model_dict.get(model_key, smp.Unet)
        model = model(encoder_name=encoder_name, classes=3, in_channels=3)
        model_list = sorted(glob.glob(path_dir + '/*_model.pkl'), key=os.path.getmtime)
        model.load_state_dict(torch.load(model_list[-1], map_location="cpu"))
    model=model.to(device)
    model.train(False)
    with torch.no_grad():
        for i,(x,) in enumerate(val_dataloader):
            if torch.cuda.is_available():
                x=x.to(device)
            y_pred=torch.softmax(model(x),1).detach().cpu().numpy()
            predictions.append(y_pred)
    predictions=np.concatenate(predictions,axis=0)
    return predictions

# Switch from plain CrossEntropyLoss to a combo loss function that combines Dice loss and CrossEntropy loss.
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1)
        targets_one_hot = nn.functional.one_hot(targets, num_classes=preds.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        intersection = (preds * targets_one_hot).sum(dim=(2,3))
        union = preds.sum(dim=(2,3)) + targets_one_hot.sum(dim=(2,3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


# Combined Dice + Cross Entropy 
class ComboLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight=None):

        super().__init__()
        if weight is not None:
            self.ce = nn.CrossEntropyLoss(weight=weight)
        else:
            self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.weight_dice = weight_dice

    def forward(self, preds, targets):
        return self.weight_dice * self.dice(preds, targets) + (1 - self.weight_dice) * self.ce(preds, targets)
