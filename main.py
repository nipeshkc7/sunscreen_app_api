from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import torch
from torchvision import transforms, models

# -------------------- General Libraries --------------------
import pandas as pd
import numpy as np
import os
import warnings
from PIL import Image

# -------------------- OpenCV Modules ---------------------
import cv2


# -------------------- PyTorch Modules --------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, models

# Initialize FastAPI app
app = FastAPI()

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Contracting path (Encoder) with BatchNorm
        self.encoder1 = self.double_conv(3, 32)
        self.encoder2 = self.double_conv(32, 64)
        self.encoder3 = self.double_conv(64, 128)
        self.encoder4 = self.double_conv(128, 256)
        self.encoder5 = self.double_conv(256, 512)

        # Maxpool layers for downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Expanding path (Decoder) with up-convolutions and skip connections
        self.upconv1 = self.up_conv(512, 256)
        self.decoder1 = self.double_conv(512, 256)
        self.upconv2 = self.up_conv(256, 128)
        self.decoder2 = self.double_conv(256, 128)
        self.upconv3 = self.up_conv(128, 64)
        self.decoder3 = self.double_conv(128, 64)
        self.upconv4 = self.up_conv(64, 32)
        self.decoder4 = self.double_conv(64, 32)

        # Final 1x1 convolution to map to 1 output channel (binary mask)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.maxpool(enc1))
        enc3 = self.encoder3(self.maxpool(enc2))
        enc4 = self.encoder4(self.maxpool(enc3))
        bottleneck = self.encoder5(self.maxpool(enc4))

        dec1 = self.upconv1(bottleneck)
        dec1 = torch.cat((enc4, dec1), dim=1)
        dec1 = self.decoder1(dec1)

        dec2 = self.upconv2(dec1)
        dec2 = torch.cat((enc3, dec2), dim=1)
        dec2 = self.decoder2(dec2)

        dec3 = self.upconv3(dec2)
        dec3 = torch.cat((enc2, dec3), dim=1)
        dec3 = self.decoder3(dec3)

        dec4 = self.upconv4(dec3)
        dec4 = torch.cat((enc1, dec4), dim=1)
        dec4 = self.decoder4(dec4)

        out = self.final_conv(dec4)
        return out


def dilate_mask(mask, dilation_iterations=1):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(mask, kernel, iterations=dilation_iterations)

def inpaint_white_regions(input_image, mask, inpaint_radius=3, method=cv2.INPAINT_TELEA):
    input_image_np = (input_image * 255).astype(np.uint8)
    dilated_mask = dilate_mask(mask, dilation_iterations=1)
    inpainted_image_np = cv2.inpaint(input_image_np, dilated_mask, inpaint_radius, method)
    return inpainted_image_np.astype(np.float32) / 255.0

def crop_and_resize_largest_region(original_image, predicted_mask, output_shape=(1, 3, 224, 224)):
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None  # Return None if no contours are found

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_region = original_image[y:y+h, x:x+w]

    if cropped_region.dtype != np.uint8:
        cropped_region = (cropped_region * 255).astype(np.uint8)

    resized_image = cv2.resize(cropped_region, (224, 224))

    # Convert to tensor and normalize to [0, 1]
    resized_tensor = torch.tensor(resized_image.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    return resized_tensor  # Output as PyTorch tensor in shape [1, 3, 224, 224]


# Define the process_image_pipeline function
def process_image_pipeline(input_image_path, config):
    """
    Processes an image by removing hair, cropping the lesion, and classifying it as benign or malignant.

    Parameters:
        input_image_path (str): Path to the input image.
        config (dict): Configuration dictionary with model parameters.

    Returns:
        None: Displays images and prints predictions.
    """

    # Load the input image
    input_image = Image.open(input_image_path).convert('RGB')
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    image_tensor = transform(input_image).unsqueeze(0).to(config["device"])
    input_image_np = image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    # Load hair removal and lesion cropping models
    hair_removal_model = UNet().to(config["device"])
    lesion_cropping_model = UNet().to(config["device"])
    hair_removal_model.load_state_dict(torch.load('unet_hair_removal_updated.pth'))
    lesion_cropping_model.load_state_dict(torch.load('unet_lesion_cropping_model.pth'))
    hair_removal_model.eval()
    lesion_cropping_model.eval()

    # Hair removal
    with torch.no_grad():
        hair_output = hair_removal_model(image_tensor)
        hair_mask = (torch.sigmoid(hair_output) > 0.5).float()
    mask_np = hair_mask.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
    inpainted_image_np = inpaint_white_regions(input_image_np, mask_np, inpaint_radius=5)
    inpainted_image_tensor = torch.tensor(inpainted_image_np.transpose(2, 0, 1)).unsqueeze(0).to(config["device"])

    # Lesion cropping
    with torch.no_grad():
        lesion_output = lesion_cropping_model(inpainted_image_tensor)
        lesion_mask = (torch.sigmoid(lesion_output) > 0.5).float()
    lesion_mask_np = lesion_mask.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
    cropped_img_np = crop_and_resize_largest_region(inpainted_image_np, lesion_mask_np)

    # Convert cropped image to tensor for prediction
    preprocess = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    cropped_img_np = cropped_img_np.squeeze(0).permute(1, 2, 0).cpu().numpy()

    cropped_image_tensor = preprocess(Image.fromarray((cropped_img_np * 255).astype(np.uint8)))
    input_batch = cropped_image_tensor.unsqueeze(0).to(config["device"])

    # Load classification model
    model = models.mobilenet_v2(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load("MobileNet_hair_removal_cropping.pth", map_location=config["device"]))
    model = model.to(config["device"])
    model.eval()

    # Classification prediction
    with torch.no_grad():
        output = model(input_batch)
        pred = output.item()
        pred_label = "Malignant" if pred >= 0.5 else "Benign"

    return pred_label

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    conf = {
    "seed": 42,
    "epochs": 50,
    "early_stopping": 10,
    "img_size": 224,
    "dataset_path": r"C:\path\to\data",
    "metadata_path": r"C:\path\to\metadata",
    "batch_size": 50,
    "learning_rate": 1e-4,
    "min_lr": 1e-6,
    "weight_decay": 1e-6,
    "loss_function": nn.BCEWithLogitsLoss,
    "optimizer": optim.Adam,
    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "scheduler_factor": 0.1,
    "scheduler_patience": 3,
    "batch_print_interval": 200,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    prediction_val = process_image_pipeline(image_bytes, conf)


    return {"prediction": prediction_val}
