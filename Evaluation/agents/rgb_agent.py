import os
from copy import deepcopy
import numpy as np
from collections import deque

import carla
from leaderboard.autoagents import autonomous_agent

from PIL import Image
import cv2
import torch

from config import RGB_Config
from data_pipeline.data_preprocessing import preprocessing

def get_entry_point():
    return 'HybridAgent'


class HybridAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file, route_index=None):
        self.track = autonomous_agent.Track.SENSORS
        self.config_path = path_to_conf_file
        self.step = -1
        self.initialized = False
        self.debug_counter = 0
        self.config = RGB_Config
        self.gps_buffer = deque(maxlen=self.config.gps_buffer_max_len) # Stores the last x updated gps signals.

        # Load model architecture and weights
        from models.resnet_rgb.architectures_v3 import Resnet_Baseline_V3_Dropout
        net = Resnet_Baseline_V3_Dropout(0.25)
        root = os.path.join(os.getenv("GITLAB_ROOT"),
                            "models", "resnet_rgb", "weights","Resnet_RGB_V3")
        net.load_state_dict(torch.load(os.path.join(root, "resnet_baseline_v3_dropout_ep10.pt")))
        self.net = net.cuda()

        # Inertia Problem variables
        self.stuck_detector = 0
        self.forced_move = 0

    def _init(self):
        self._route_planner = RoutePlanner(self.config.route_planner_min_distance, self.config.route_planner_max_distance)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    def sensors(self):
        sensors = [
                    {
                        'type': 'sensor.camera.rgb',
                        'x': self.config.camera_pos[0], 'y': self.config.camera_pos[1], 'z':self.config.camera_pos[2],
                        'roll': self.config.camera_rot_0[0], 'pitch': self.config.camera_rot_0[1], 'yaw': self.config.camera_rot_0[2],
                        'width': self.config.camera_width, 'height': self.config.camera_height, 'fov': self.config.camera_fov,
                        'id': 'rgb_front'
                        },
                    {
                        'type': 'sensor.camera.rgb',
                        'x': self.config.camera_pos[0], 'y': self.config.camera_pos[1], 'z':self.config.camera_pos[2],
                        'roll': self.config.camera_rot_1[0], 'pitch': self.config.camera_rot_1[1], 'yaw': self.config.camera_rot_1[2],
                        'width': self.config.camera_width, 'height': self.config.camera_height, 'fov': self.config.camera_fov,
                        'id': 'rgb_left'
                        },
                    {
                        'type': 'sensor.camera.rgb',
                        'x': self.config.camera_pos[0], 'y': self.config.camera_pos[1], 'z':self.config.camera_pos[2],
                        'roll': self.config.camera_rot_2[0], 'pitch': self.config.camera_rot_2[1], 'yaw': self.config.camera_rot_2[2],
                        'width': self.config.camera_width, 'height': self.config.camera_height, 'fov': self.config.camera_fov,
                        'id': 'rgb_right'
                        },
                    {
                        'type': 'sensor.other.imu',
                        'x': 0.0, 'y': 0.0, 'z': 0.0,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                        'sensor_tick': self.config.carla_frame_rate,
                        'id': 'imu'
                        },
                    {
                        'type': 'sensor.other.gnss',
                        'x': 0.0, 'y': 0.0, 'z': 0.0,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                        'sensor_tick': 0.01,
                        'id': 'gps'
                        },
                    {
                        'type': 'sensor.speedometer',
                        'reading_frequency': self.config.carla_fps,
                        'id': 'speed'
                        }
                    ]


        return sensors

    # Processes the images like training dataset
    def scale_crop(self, image, scale=1, start_x=0, crop_x=None, start_y=0, crop_y=None):
        (width, height) = (image.width // scale, image.height // scale)
        if scale != 1:
            image = image.resize((width, height))
        if crop_x is None:
            crop_x = width
        if crop_y is None:
            crop_y = height

        image = np.asarray(image)
        cropped_image = image[start_y:start_y + crop_y, start_x:start_x + crop_x]

        return cropped_image

    def tick(self, input_data): # Prepares data to be processed during run_step

        # IMAGE PROCESSING
        rgb = []
        for pos in ['left', 'front', 'right']:
            rgb_cam = 'rgb_' + pos
            rgb_pos = cv2.cvtColor(input_data[rgb_cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            rgb_pos = self.scale_crop(Image.fromarray(rgb_pos), self.config.scale, self.config.img_width, self.config.img_width, self.config.img_resolution[0], self.config.img_resolution[0])
            rgb.append(rgb_pos)
        rgb = np.concatenate(rgb, axis=1)

        # NAVIGATION
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']

        compass = input_data['imu'][1][-1]
        if (np.isnan(compass) == True): # CARLA 0.9.10 occasionally sends NaN values in the compass
            compass = 0.0

        result = {
                'rgb': rgb,
                'gps': gps,
                'speed': speed,
                'compass': compass,
                }

        pos = self._get_position(result)
        result['gps'] = pos

        self.gps_buffer.append(pos)
        denoised_pos = np.average(self.gps_buffer, axis=0)
        waypoint_route = self._route_planner.run_step(denoised_pos)
        next_wp, next_cmd = waypoint_route[0] # waypoint_route[1] if len(waypoint_route) > 1 else waypoint_route[0]
        result['next_command'] = next_cmd.value

        print(str(next_cmd))
        print("\n")

        return result



    @torch.inference_mode() # Faster version of torch_no_grad
    def run_step(self, input_data, timestamp):
        self.step += 1

        if not self.initialized:
            self._init()
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            self.control = control        

        tick_data = self.tick(input_data)

        # INERTIA
        is_stuck = False
        if(self.stuck_detector > self.config.stuck_threshold and self.forced_move < self.config.creep_duration):
            print("Detected agent being stuck. Move for frame: ", self.forced_move)
            is_stuck = True
            self.forced_move += 1

        if self.forced_move == self.config.creep_duration:
            self.forced_move = 0
            self.stuck_detector = 0

        ### PREPROCESSING

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # RGB
        img = tick_data['rgb'] # 160,960,3
        img_batch = torch.unsqueeze(torch.tensor(img), dim=0).transpose(1, 3).transpose(2, 3).float() #1, 3, 160, 960
        img_norm = preprocessing["rgb"](img_batch).float()

        #  NAVIGATION
        cmd_labels = torch.tensor(tick_data['next_command'])
        cmd_one_hot = preprocessing["command"](cmd_labels).float()

        # SPEED
        spd = torch.tensor(tick_data['speed'])
        spd_norm = preprocessing["speed"](spd).float()

        #### FORWARD PASS,
        img_norm = img_norm.to(device)
        cmd_one_hot = torch.unsqueeze(cmd_one_hot.to(device),0)
        spd_norm = torch.unsqueeze(torch.unsqueeze(spd_norm.to(device), 0),0)

        with torch.no_grad():
            self.net.eval()
            outputs_ = self.net(img_norm,cmd_one_hot,spd_norm)
        brake, steer, throttle = outputs_
        print(brake)

        ### INTERIA STEER MODULATION

        if (tick_data['speed'] < 0.5):  # 0.1 is just an arbitrary low number to threshhold when the car is stopped
            self.stuck_detector += 1
        elif (tick_data['speed'] > 0.5 and is_stuck == False):
            self.stuck_detector = 0
            self.forced_move = 0

        ### CERATE CARLA CONTROLS
        control = carla.VehicleControl()

        if is_stuck:
            control.throttle = 0.5
            control.steer = float(steer)
            control.brake = 0
        else:
            control.throttle = float(throttle)
            if brake > 0.01:
                control.brake = float(brake)
            else:
                control.brake = float(0)
            control.steer = float(steer)
        self.control = control

        return control

# Taken from LBC (Learning by cheating) https://arxiv.org/abs/1912.12294
class RoutePlanner(object):
    def __init__(self, min_distance, max_distance):
        self.saved_route = deque()
        self.route = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.is_last = False

        self.mean = np.array([0.0, 0.0]) # for carla 9.10
        self.scale = np.array([111324.60662786, 111319.490945]) # for carla 9.10

    def set_route(self, global_plan, gps=False):
        self.route.clear()

        for pos, cmd in global_plan:
            if gps:
                pos = np.array([pos['lat'], pos['lon']])
                pos -= self.mean
                pos *= self.scale
            else:
                pos = np.array([pos.location.x, pos.location.y])
                pos -= self.mean

            self.route.append((pos, cmd))

    def run_step(self, gps):
        if len(self.route) <= 2:
            self.is_last = True
            return self.route

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i-1][0])
            distance = np.linalg.norm(self.route[i][0] - gps)

            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()

        return self.route

    def save(self):
        self.saved_route = deepcopy(self.route)

    def load(self):
        self.route = self.saved_route
        self.is_last = False

