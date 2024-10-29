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

# Define the UNet model for hair removal and lesion cropping
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the models and pre-trained weights
hair_removal_model = UNet().to(device)
lesion_cropping_model = UNet().to(device)
classification_model = models.mobilenet_v2(pretrained=True).to(device)

hair_removal_model.load_state_dict(torch.load('unet_hair_removal_updated.pth', map_location=device))
lesion_cropping_model.load_state_dict(torch.load('unet_lesion_cropping_model.pth', map_location=device))
classification_model.load_state_dict(torch.load("MobileNet_hair_removal_cropping.pth", map_location=device))

# Image preprocessing
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    return transform(img).unsqueeze(0).to(device)

# Image inpainting and cropping functions
def dilate_mask(mask, dilation_iterations=1):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(mask, kernel, iterations=dilation_iterations)

def inpaint_white_regions(input_image_np, mask, inpaint_radius=3, method=cv2.INPAINT_TELEA):
    input_image_np = (input_image_np * 255).astype(np.uint8)
    dilated_mask = dilate_mask(mask, dilation_iterations=1)
    inpainted_image_np = cv2.inpaint(input_image_np, dilated_mask, inpaint_radius, method)
    return inpainted_image_np.astype(np.float32) / 255.0

def crop_and_resize_largest_region(original_image_np, predicted_mask, output_shape=(1, 3, 224, 224)):
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None  # No contours found

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_region = original_image_np[y:y+h, x:x+w]
    resized_image = cv2.resize(cropped_region, (224, 224))
    resized_tensor = torch.tensor(resized_image.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    return resized_tensor

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_image = preprocess_image(image_bytes)

    # Step 1: Hair removal
    with torch.no_grad():
        hair_output = hair_removal_model(input_image)
        hair_mask = (torch.sigmoid(hair_output) > 0.5).float().cpu().numpy().squeeze()
        input_image_np = input_image.cpu().squeeze().numpy().transpose(1, 2, 0)
        inpainted_image_np = inpaint_white_regions(input_image_np, hair_mask)
        inpainted_image_tensor = torch.tensor(inpainted_image_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # Step 2: Lesion cropping
    with torch.no_grad():
        lesion_output = lesion_cropping_model(inpainted_image_tensor)
        lesion_mask = (torch.sigmoid(lesion_output) > 0.5).float().cpu().numpy().squeeze()
        cropped_image_tensor = crop_and_resize_largest_region(inpainted_image_np, lesion_mask)

    # Step 3: Classification prediction
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cropped_image = Image.fromarray((cropped_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    final_input = preprocess(cropped_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = classification_model(final_input)
        pred = output.item()
        pred_label = "Malignant" if pred >= 0.5 else "Benign"

    return {"prediction": pred, "label": pred_label}
