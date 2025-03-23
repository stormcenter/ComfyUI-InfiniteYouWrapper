# Copyright (c) 2025 (ComfyUI Integration for InfiniteYou)
# Based on InfiniteYou by Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

import math
import numpy as np
import cv2
import torch
from PIL import Image
from insightface.utils import face_align
from facexlib.recognition import init_recognition_model


def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    """
    Draw facial keypoints on the image.
    
    Args:
        image_pil (PIL.Image): Input image
        kps (np.ndarray): Facial keypoints coordinates
        color_list (list): List of RGB color tuples for different keypoints
        
    Returns:
        PIL.Image: Image with keypoints drawn
    """
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    # Draw lines connecting keypoints
    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    
    # Make the lines semi-transparent
    out_img = (out_img * 0.6).astype(np.uint8)

    # Draw keypoints
    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil


def extract_arcface_embedding(in_image, landmark, arcface_model=None):
    """
    Extract ArcFace embedding from a face image.
    
    Args:
        in_image (np.ndarray): BGR format input image
        landmark (np.ndarray): Facial landmark coordinates
        arcface_model: Pre-loaded ArcFace model or None to load a new one
        
    Returns:
        torch.Tensor: Face embedding vector (512 dimensions, normalized)
    """
    kps = landmark
    # Normalize crop the face according to the landmarks
    arc_face_image = face_align.norm_crop(in_image, landmark=np.array(kps), image_size=112)
    
    # Convert to tensor and normalize
    arc_face_image = torch.from_numpy(arc_face_image).unsqueeze(0).permute(0,3,1,2) / 255.
    arc_face_image = 2 * arc_face_image - 1
    arc_face_image = arc_face_image.cuda().contiguous()
    
    # Load model if not provided
    if arcface_model is None:
        arcface_model = init_recognition_model('arcface', device='cuda')
    
    # Extract face embedding
    face_emb = arcface_model(arc_face_image)[0]  # [512], normalized
    return face_emb


def resize_and_pad_image(source_img, target_img_size):
    """
    Resize an image to fit within target dimensions while maintaining aspect ratio,
    then pad with white to reach target size.
    
    Args:
        source_img (PIL.Image): Input image
        target_img_size (tuple): Target width and height as (width, height)
        
    Returns:
        PIL.Image: Resized and padded image
    """
    # Get original and target sizes
    source_img_size = source_img.size
    target_width, target_height = target_img_size
    
    # Determine the new size based on the shorter side of target_img
    if target_width <= target_height:
        new_width = target_width
        new_height = int(target_width * (source_img_size[1] / source_img_size[0]))
    else:
        new_height = target_height
        new_width = int(target_height * (source_img_size[0] / source_img_size[1]))
    
    # Resize the source image using LANCZOS interpolation for high quality
    resized_source_img = source_img.resize((new_width, new_height), Image.LANCZOS)
    
    # Compute padding to center resized image
    pad_left = (target_width - new_width) // 2
    pad_top = (target_height - new_height) // 2
    
    # Create a new image with white background
    padded_img = Image.new("RGB", target_img_size, (255, 255, 255))
    padded_img.paste(resized_source_img, (pad_left, pad_top))
    
    return padded_img


def detect_face(face_analyzers, image_cv2):
    """
    Detect faces in an image using multiple scales if needed.
    
    Args:
        face_analyzers (dict): Dictionary containing face analyzers at different scales
        image_cv2 (np.ndarray): BGR format input image
        
    Returns:
        list: List of detected face information
    """
    # Try with largest detector first
    face_info = face_analyzers["app_640"].get(image_cv2)
    if len(face_info) > 0:
        return face_info
    
    # Try with medium detector if no faces found
    face_info = face_analyzers["app_320"].get(image_cv2)
    if len(face_info) > 0:
        return face_info
    
    # Try with smallest detector as last resort
    face_info = face_analyzers["app_160"].get(image_cv2)
    return face_info


def get_largest_face(face_info):
    """
    Get the largest face from a list of detected faces.
    
    Args:
        face_info (list): List of detected face information
        
    Returns:
        dict: Information about the largest face
    """
    if len(face_info) == 0:
        return None
    
    # Sort faces by area (width * height) and return the largest
    largest_face = sorted(
        face_info, 
        key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1])
    )[-1]
    
    return largest_face


def create_control_image(height, width):
    """
    Create a blank black control image.
    
    Args:
        height (int): Height of the image
        width (int): Width of the image
        
    Returns:
        PIL.Image: Black image of specified dimensions
    """
    out_img = np.zeros([height, width, 3])
    control_image = Image.fromarray(out_img.astype(np.uint8))
    return control_image


def process_id_image(image, face_analyzers, arcface_model, image_proj_model):
    """
    Process an identity image to extract embeddings.
    
    Args:
        image (PIL.Image): Input identity image
        face_analyzers (dict): Dictionary of face analyzers
        arcface_model: ArcFace recognition model
        image_proj_model: Image projection model
        
    Returns:
        tuple: (face_info, id_embed) detected face info and identity embedding
    """
    # Convert to CV2 format
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Detect face
    face_info = detect_face(face_analyzers, image_cv2)
    if len(face_info) == 0:
        raise ValueError('No face detected in the input ID image')
    
    # Get largest face
    face_info = get_largest_face(face_info)
    landmark = face_info['kps']
    
    # Extract embedding
    id_embed = extract_arcface_embedding(image_cv2, landmark, arcface_model)
    id_embed = id_embed.clone().unsqueeze(0).float().cuda()
    id_embed = id_embed.reshape([1, -1, 512])
    id_embed = id_embed.to(device='cuda', dtype=torch.bfloat16)
    
    # Process through image projection model
    with torch.no_grad():
        id_embed = image_proj_model(id_embed)
        bs_embed, seq_len, _ = id_embed.shape
        id_embed = id_embed.view(bs_embed, seq_len, -1)
        id_embed = id_embed.to(device='cuda', dtype=torch.bfloat16)
    
    return face_info, id_embed


def process_control_image(control_image, width, height, face_analyzers):
    """
    Process a control image for facial keypoint extraction.
    
    Args:
        control_image (PIL.Image): Input control image
        width (int): Target width
        height (int): Target height
        face_analyzers (dict): Dictionary of face analyzers
        
    Returns:
        PIL.Image: Processed control image with facial keypoints
    """
    if control_image is None:
        return create_control_image(height, width)
    
    control_image = control_image.convert("RGB")
    control_image = resize_and_pad_image(control_image, (width, height))
    
    # Detect face in control image
    control_face_info = detect_face(face_analyzers, cv2.cvtColor(np.array(control_image), cv2.COLOR_RGB2BGR))
    
    if len(control_face_info) == 0:
        print("Warning: No face detected in control image. Using black image instead.")
        return create_control_image(height, width)
    
    # Get largest face and draw keypoints
    control_face_info = get_largest_face(control_face_info)
    control_image_processed = draw_kps(control_image, control_face_info['kps'])
    
    return control_image_processed