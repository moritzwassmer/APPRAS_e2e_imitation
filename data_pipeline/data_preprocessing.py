from torchvision import transforms
import torch
import numpy as np

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
    transforms.Normalize(mean=[105.6161, 81.5673, 79.6657], std=[66.2220, 60.1001, 66.8309]), # changed to rgb
    ])(X_rgb)

def prep_speed(X_spd):
    # speed_mean: 2.382234  old 2.250456762830466
    # speed_std: 1.724884  old 0.3021584025489131
    return ((X_spd - 2.382234)/ 1.724884)
    
def prep_command(X_cmd):
    X_cmd = torch.where(X_cmd == -1, torch.tensor(0, dtype=X_cmd.dtype), X_cmd).to(torch.int64) # Replace by -1 by 0
    X_cmd = torch.nn.functional.one_hot(X_cmd, num_classes=7)
    return torch.squeeze(X_cmd)

def transform_lidar_bev(points, sr=(-16,16),fr=(0,32),hr=(-2,1),res = 0.05):
    side_range = sr    # left-most to right-most
    fwd_range =fr   # back-most to forward-most

    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[:, 1]
    y_points = points[:, 0]
    z_points = points[:, 2]

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]


    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-x_points / res).astype(np.int32)  # y axis is -x in LIDAR


    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor and ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))


    height_range = hr  # bottom-most to upper-most

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a = z_points,
                           a_min=height_range[0],
                           a_max=height_range[1])

    def scale_to_255(a, min, max, dtype=np.uint8):
    #Scales an array of values from specified min, max range to 0-255
    #Optionally specify the data type of the output (default is uint8)
   
        return (((a - min) / float(max - min)) * 255).astype(dtype)

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = scale_to_255(pixel_values, min=height_range[0], max=height_range[1])


    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 1+int((side_range[1] - side_range[0])/res)
    y_max = 1+int((fwd_range[1] - fwd_range[0])/res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    im[y_img, x_img] = pixel_values

    return im



preprocessing = {
    "rgb": prep_rgb, 
    "speed": prep_speed,
    "command": prep_command,
    "lidar": transform_lidar_bev
}
