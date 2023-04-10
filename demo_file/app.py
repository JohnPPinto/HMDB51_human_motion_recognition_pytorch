import gradio as gr
import os
import torch
import torchvision
from timeit import default_timer as timer
from model import create_model
from utils import preprocess_video

# Configuring the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setting class names using the text file
with open('class_names.txt', 'r') as f:
    class_names = [i.strip() for i in f.readlines()]

# Setting the model and transforms
model, transforms = create_model(num_classes=len(class_names), seed=42)
model.load_state_dict(torch.load(f='mvit_v2_pretrained_model_hmdb51.pth', map_location=device))

# Creating a function to predict the video
def predict(video_file):
    """
    A function to predict the video using the model.
    Parameters: 
        video_file: str, A video file path as a string.
    Returns: 
        pred_labels_probs: A dict file containing the class names and confidence values.
        pred_time: Time taken to predict in seconds.
    """
    # Preprocessing the video file
    frames = preprocess_video(video=video_file)
    
    # transforming the frames
    mod_frames = transforms(frames).unsqueeze(dim=0)
    
    # Starting the timer and predicting using the model
    start_time = timer()
    model.eval()
    
    # forward pass
    with torch.no_grad():
        logits = model(mod_frames.to(device)) 
        pred_probs = torch.softmax(logits, dim=1).squeeze(dim=0)
    
    # Creating a dict with class names and their predicted probabilities
    pred_labels_probs = {class_names[i]: float(pred_probs[i]) for i in range(len(class_names))}
        
    # Ending the timer and calculating the prediction time.
    end_time = timer()
    pred_time = round(end_time - start_time, 5)
    
    # Returning the labels and time for gradio output.
    return pred_labels_probs, pred_time

# Pre-information for interface
title = 'Human Activity Recognition(HAR)'
description = 'A Gradio demo web application for video classification using the MViT V2 Pretrained Model and trained on the HMDB51 Dataset.'
article = 'Created by John'
example_list = [['examples/'+ i] for i in os.listdir('examples')]

# Building the Gradio interface
demo = gr.Interface(fn=predict,
                    inputs=gr.Video(label='Video'),
                    outputs=[gr.Label(num_top_classes=5, label='Predictions'),
                             gr.Number(label='Prediction Time (sec)')],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

# Launching the gradio interface
demo.launch()
