import torch
import clip
from PIL import Image
import torch.nn.functional as F
import time
# from utils import *
from tqdm import tqdm
from CLIP_utils import *
import matplotlib.pyplot as plt
import numpy as np

seed = 1  # if seed is needed
set_random_seed(seed)

class CLIP_LoRA(nn.Module):
    def __init__(self, e_weight_path, b_weight_path, device="cuda"):
        super(CLIP_LoRA, self).__init__()
        self.device = device
        self.model_e, self.preprocess = clip.load("ViT-B/16", device=self.device)
        self.model_b, self.preprocess = clip.load("ViT-B/16", device=self.device)
        self.e_args = get_arguments()
        self.e_args.weight_path = e_weight_path
        self.e_list_lora_layers = apply_lora(self.e_args, self.model_e)
        load_lora(self.e_args, self.e_list_lora_layers)
        
        # Load arguments and weights for base
        self.b_args = get_arguments()
        self.b_args.weight_path = b_weight_path
        self.b_list_lora_layers = apply_lora(self.b_args, self.model_b)
        load_lora(self.b_args, self.b_list_lora_layers)
        
        self.model_e.to(self.device)
        self.model_b.to(self.device)
        
    def forward(self, img, e_captions, b_captions, from_file=True):
        e_text = clip.tokenize(e_captions).to(self.device)
        b_text = clip.tokenize(b_captions).to(self.device)
        if from_file:
            image = self.preprocess(Image.open(img)).unsqueeze(0).to(self.device)
        else:
            image = self.preprocess(img).unsqueeze(0).to(self.device)
        results = {"end-effector": False, "base": False}
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # Inference for end-effector
                logits_per_image, logits_per_text = self.model_e(image, e_text)
                self.e_probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                # print(e_probs)
                results["end-effector"] = True if np.argmax(self.e_probs) == 1 else False
                
                # Inference for base
                logits_per_image, logits_per_text = self.model_b(image, b_text)
                self.b_probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                results["base"] = True if np.argmax(self.b_probs) == 1 else False
                # print(self.e_probs)
                
        return results
    

def CLIP_inference(img_path, e_filter_confidence, b_filter_confidence):
    e_weight_path = 'CLIP_logs/vitb16/robotgripper_test/32shots/seed1/lora_weights.pt'
    b_weight_path = 'CLIP_logs/vitb16/robotbase_test/32shots/seed1/lora_weights.pt'
    e_caps = ["a photo without robot end-effector", "a photo of robot end-effector"]
    b_caps = ["a photo without robot base", "a photo of robot base"] 
    model = CLIP_LoRA(e_weight_path, b_weight_path)
    result = model(img_path, e_caps, b_caps)
    if model.e_probs.squeeze()[0] >= e_filter_confidence:
        result["end-effector"] = False
    if model.b_probs.squeeze()[0] >= b_filter_confidence:
        result["base"] = False
    return result

if __name__ == "__main__":
    img_path = 'frames_dir19/frame_00001.jpg'
    e_weight_path = 'logs/vitb16/robotgripper_test/32shots/seed1/lora_weights.pt'
    b_weight_path = 'logs/vitb16/robotbase_test/32shots/seed1/lora_weights.pt'
    e_caps = ["a photo without robot end-effector", "a photo of robot end-effector"]
    b_caps = ["a photo without robot base", "a photo of robot base"]

    model = CLIP_LoRA(e_weight_path, b_weight_path)
    result = model(img_path, e_caps, b_caps)

    print(result)
    