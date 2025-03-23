import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2
import math
import torch.nn as nn
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from facexlib.recognition import init_recognition_model
import folder_paths

# Add local InfiniteYou project to path to import its modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "InfiniteYou"))

# Import Resampler model
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.GELU(),
            nn.Linear(inner_dim, dim, bias=False),
        )
    
    def forward(self, x):
        return self.net(x)

def reshape_tensor(x, heads):
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    x = x.reshape(bs, heads, length, -1)
    return x

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm1(x)
        latents = self.norm2(latents)
        
        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)

class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
    ):
        super().__init__()
        
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        
        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        latents = self.latents.repeat(x.size(0), 1, 1)
        
        x = self.proj_in(x)
        
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
            
        latents = self.proj_out(latents)
        return self.norm_out(latents)

# Define utility functions from InfiniteYou
def resize_and_pad_image(source_img, target_img_size):
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

def extract_arcface_embedding(in_image, landmark, arcface_model=None):
    kps = landmark
    arc_face_image = face_align.norm_crop(in_image, landmark=np.array(kps), image_size=112)
    arc_face_image = torch.from_numpy(arc_face_image).unsqueeze(0).permute(0,3,1,2) / 255.
    arc_face_image = 2 * arc_face_image - 1
    arc_face_image = arc_face_image.cuda().contiguous()
    if arcface_model is None:
        arcface_model = init_recognition_model('arcface', device='cuda')
    face_emb = arcface_model(arc_face_image)[0]  # [512], normalized
    return face_emb

def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil


# Define ComfyUI node for loading InfiniteYou model
class LoadInfuModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "./models/InfiniteYou"}),
                "infu_flux_version": (["v1.0"], {"default": "v1.0"}),
                "model_version": (["aes_stage2", "sim_stage1"], {"default": "aes_stage2"}),
            },
        }

    RETURN_TYPES = ("INFU_MODEL",)
    RETURN_NAMES = ("infu_model",)
    FUNCTION = "load_model"
    CATEGORY = "InfiniteYou"

    def load_model(self, model_path, infu_flux_version, model_version):
        # Prepare paths
        infu_model_path = os.path.join(model_path, f"infu_flux_{infu_flux_version}", model_version)
        infusenet_path = os.path.join(infu_model_path, 'InfuseNetModel')
        image_proj_model_path = os.path.join(infu_model_path, 'image_proj_model.bin')
        insightface_root_path = os.path.join(model_path, 'supports', 'insightface')
        
        # Load InsightFace models
        app_640 = FaceAnalysis(name='antelopev2', 
                          root=insightface_root_path, 
                          providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app_640.prepare(ctx_id=0, det_size=(640, 640))
        
        app_320 = FaceAnalysis(name='antelopev2', 
                          root=insightface_root_path, 
                          providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app_320.prepare(ctx_id=0, det_size=(320, 320))
        
        app_160 = FaceAnalysis(name='antelopev2', 
                          root=insightface_root_path, 
                          providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app_160.prepare(ctx_id=0, det_size=(160, 160))
        
        # Load ArcFace model
        arcface_model = init_recognition_model('arcface', device='cuda')
        
        # Load image projection model
        num_tokens = 8  # image_proj_num_tokens
        image_emb_dim = 512
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=num_tokens,
            embedding_dim=image_emb_dim,
            output_dim=4096,
            ff_mult=4,
        )
        
        ipm_state_dict = torch.load(image_proj_model_path, map_location="cpu")
        image_proj_model.load_state_dict(ipm_state_dict['image_proj'])
        image_proj_model.to('cuda', torch.bfloat16)
        image_proj_model.eval()
        
        # No need to load the ControlNet in this node, as it will be handled by the FLUX pipeline
        
        # Pack everything into a model dict
        infu_model = {
            "face_analyzers": {
                "app_640": app_640,
                "app_320": app_320,
                "app_160": app_160
            },
            "arcface_model": arcface_model,
            "image_proj_model": image_proj_model,
            "model_version": model_version,
            "model_path": infu_model_path,
            "infusenet_path": infusenet_path
        }
        
        print(f"InfiniteYou model loaded: {model_version}")
        return (infu_model,)


# Define ComfyUI node for applying InfiniteYou model
class ApplyInfu:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "infu_model": ("INFU_MODEL",),
                "id_image": ("IMAGE",),
                "width": ("INT", {"default": 864, "min": 512, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1152, "min": 512, "max": 2048, "step": 8}),
                "infusenet_conditioning_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "infusenet_guidance_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "infusenet_guidance_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "control_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("FLUX_CONTROLNET", "CONTROLNET_EMBEDS")
    RETURN_NAMES = ("infusenet", "id_embeds")
    FUNCTION = "process"
    CATEGORY = "InfiniteYou"

    def process(self, infu_model, id_image, width, height, infusenet_conditioning_scale, 
                infusenet_guidance_start, infusenet_guidance_end, control_image=None):
        from diffusers.models import FluxControlNetModel
        
        # Extract the components from the model dictionary
        face_analyzers = infu_model["face_analyzers"]
        arcface_model = infu_model["arcface_model"]
        image_proj_model = infu_model["image_proj_model"]
        infusenet_path = infu_model["infusenet_path"]
        
        # Load the InfuseNet ControlNet model
        infusenet = FluxControlNetModel.from_pretrained(infusenet_path, torch_dtype=torch.bfloat16)
        infusenet.to("cuda")
        
        # Convert tensor to PIL image for ID image
        if isinstance(id_image, torch.Tensor):
            # Convert from [B, H, W, C] to PIL
            if id_image.dim() == 4 and id_image.shape[0] == 1:
                id_image = id_image.squeeze(0)
            
            id_image_np = (id_image.cpu().numpy() * 255).astype(np.uint8)
            id_image_pil = Image.fromarray(id_image_np)
        else:
            id_image_pil = id_image
        
        # Process ID image to extract embeddings
        id_image_cv2 = cv2.cvtColor(np.array(id_image_pil), cv2.COLOR_RGB2BGR)
        
        # Detect face using various scales if needed
        face_info = face_analyzers["app_640"].get(id_image_cv2)
        if len(face_info) == 0:
            face_info = face_analyzers["app_320"].get(id_image_cv2)
        if len(face_info) == 0:
            face_info = face_analyzers["app_160"].get(id_image_cv2)
        
        if len(face_info) == 0:
            raise ValueError('No face detected in the input ID image')
            
        # Use the largest face
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        landmark = face_info['kps']
        
        # Extract face embedding
        id_embed = extract_arcface_embedding(id_image_cv2, landmark, arcface_model)
        id_embed = id_embed.clone().unsqueeze(0).float().cuda()
        id_embed = id_embed.reshape([1, -1, 512])
        id_embed = id_embed.to(device='cuda', dtype=torch.bfloat16)
        
        # Process through image projection model
        with torch.no_grad():
            id_embed = image_proj_model(id_embed)
            bs_embed, seq_len, _ = id_embed.shape
            id_embed = id_embed.view(bs_embed, seq_len, -1)
            id_embed = id_embed.to(device='cuda', dtype=torch.bfloat16)
        
        # Process control image if provided
        if control_image is not None:
            # Convert tensor to PIL image for control image
            if isinstance(control_image, torch.Tensor):
                if control_image.dim() == 4 and control_image.shape[0] == 1:
                    control_image = control_image.squeeze(0)
                
                control_image_np = (control_image.cpu().numpy() * 255).astype(np.uint8)
                control_image_pil = Image.fromarray(control_image_np)
            else:
                control_image_pil = control_image
            
            control_image_pil = control_image_pil.convert("RGB")
            control_image_pil = resize_and_pad_image(control_image_pil, (width, height))
            
            # Detect face in control image
            control_face_info = face_analyzers["app_640"].get(cv2.cvtColor(np.array(control_image_pil), cv2.COLOR_RGB2BGR))
            if len(control_face_info) == 0:
                print("Warning: No face detected in control image. Using black image instead.")
                out_img = np.zeros([height, width, 3])
                control_image_processed = Image.fromarray(out_img.astype(np.uint8))
            else:
                # Use the largest face
                control_face_info = sorted(control_face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
                control_image_processed = draw_kps(control_image_pil, control_face_info['kps'])
        else:
            # Create a black control image if none provided
            out_img = np.zeros([height, width, 3])
            control_image_processed = Image.fromarray(out_img.astype(np.uint8))
        
        # Convert processed control image to tensor
        control_image_tensor = torch.from_numpy(np.array(control_image_processed)).float() / 255.0
        if control_image_tensor.dim() == 3:
            control_image_tensor = control_image_tensor.unsqueeze(0)
        
        # Return the InfuseNet model, ID embeddings, and processed control image
        return (infusenet, id_embed)


# Node for creating InfiniteYou conditioning parameters
class InfuConditioningParams:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "infusenet": ("FLUX_CONTROLNET",),
                "id_embeds": ("CONTROLNET_EMBEDS",),
                "infusenet_conditioning_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "infusenet_guidance_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "infusenet_guidance_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "create_conditioning"
    CATEGORY = "InfiniteYou"

    def create_conditioning(self, infusenet, id_embeds, infusenet_conditioning_scale, 
                           infusenet_guidance_start, infusenet_guidance_end):
        # Create a conditioning dictionary that ComfyUI can understand
        conditioning = {
            "controlnet": infusenet,
            "controlnet_prompt_embeds": id_embeds,
            "controlnet_conditioning_scale": infusenet_conditioning_scale,
            "control_guidance_start": infusenet_guidance_start,
            "control_guidance_end": infusenet_guidance_end,
        }
        
        return (conditioning,)


# Node mapping
NODE_CLASS_MAPPINGS = {
    "LoadInfuModel": LoadInfuModel,
    "ApplyInfu": ApplyInfu,
    "InfuConditioningParams": InfuConditioningParams,
}

# Node display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadInfuModel": "Load InfiniteYou Model",
    "ApplyInfu": "Apply InfiniteYou",
    "InfuConditioningParams": "InfiniteYou Conditioning Parameters",
}