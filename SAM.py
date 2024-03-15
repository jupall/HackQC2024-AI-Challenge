
import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import copy
from segment_anything import sam_model_registry, SamPredictor
from os import listdir
from os.path import isfile, join

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   

def remove_background_with_SAM(images_path:str, input_box, input_point,input_label, output_path:str, sam_checkpoint:str="../sam_vit_h_4b8939.pth", model_type:str="vit_h", device:str = "cuda", grayscale:bool=True):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    

    
    images_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    count = 0
    for img in images_files:
        
        image = cv2.imread(images_path + img)
        predictor.set_image(image)
        image_copy = copy.deepcopy(image)
        
        masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=False,)
        
        image_copy[masks[0]==False] = [0,0,0] 
        
        if grayscale : 
            img_copy_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(output_path + "SAM_" + img, img_copy_gray)
        else:
            cv2.imwrite(output_path + "SAM_" + img, image_copy)
        
