from torchvision import transforms
import torch

"""
The script only contains dictionaries that define for modalities (keys) how they need to be transformed (values).
These dictionaries shall be used directly after a batch is loaded!
"""

def prep_rgb(X_rgb):
    return transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # transforms.Normalize(mean=[62.4933, 73.9556, 81.5393], std=[55.3234, 54.6214, 58.7628]),
    transforms.Normalize(mean=[79.6657, 81.5673, 105.6161], std=[66.8309, 60.1001, 66.2220]),
    ])(X_rgb)

def prep_speed(X_spd):
    # speed_mean: 2.382234  old 2.250456762830466
    # speed_std: 1.724884  old 0.3021584025489131
    return ((X_spd - 2.382234)/ 1.724884)
    
def prep_command(X_cmd):
    X_cmd = torch.where(X_cmd == -1, torch.tensor(0, dtype=X_cmd.dtype), X_cmd).to(torch.int64) # Replace by -1 by 0
    X_cmd = torch.nn.functional.one_hot(X_cmd, num_classes=7)
    return torch.squeeze(X_cmd)


preprocessing = {
    "rgb": prep_rgb, 
    "speed": prep_speed,
    "command": prep_command,
}
