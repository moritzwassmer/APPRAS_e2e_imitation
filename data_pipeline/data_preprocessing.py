from torchvision import transforms

"""
The script only contains dictionaries that define for modalities (keys) how they need to be transformed (values).
These dictionaries shall be used directly after a batch is loaded!
"""

preprocessing = {
    "rgb": transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Normalize(mean=[62.4933, 73.9556, 81.5393], std=[55.3234, 54.6214, 58.7628]),
    ])
}
