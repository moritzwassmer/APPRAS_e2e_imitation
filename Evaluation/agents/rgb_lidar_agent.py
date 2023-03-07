import os
import numpy as np
from collections import deque

import carla
from leaderboard.autoagents import autonomous_agent

from PIL import Image
import cv2
import torch

from config import Lidar_Config
from data_pipeline.data_preprocessing import preprocessing, transform_lidar_bev
import utils

def get_entry_point():
    return 'HybridAgent'


class HybridAgent(autonomous_agent.AutonomousAgent):

    """Defines the Agent to be run from the leaderboard

    Attributes:
        track: Sensors or Map track of Leaderboard
        config_path: path to config file
        step: Integer of current step of simulation
        initialized: Boolean whether initialized or not
        config: config class describing sensor and carla settings
        gps_buffer: Deque which stores last GPS positions
        net: pytorch network
        stuck_detector: Counter for how long agent didn't move
        forced_move: Counter for how many steps the car moved when it was detected being stuck
        _route_planner: RoutePlanner for navigation
    """

    def setup(self, path_to_conf_file, route_index=None):
        """Sets the agent up and initialized most attributes

        Args:
            path_to_conf_file: String path to config file
            route_index: index of route which is run from leaderboard
        """

        self.track = autonomous_agent.Track.SENSORS
        self.config_path = path_to_conf_file
        self.step = -1
        self.initialized = False
        self.config = Lidar_Config
        self.gps_buffer = deque(maxlen=self.config.gps_buffer_max_len) # Stores the last x updated gps signals.
        self.lidar_pos = self.config.lidar_pos  # x, y, z coordinates of the LiDAR position.

        # Load model architecture and weights
        from models.resnet_lidar.lidar_v1 import Resnet_Lidar_V1_Dropout
        net = Resnet_Lidar_V1_Dropout(0.25)
        root = os.path.join(os.getenv("GITLAB_ROOT"),
                            "models", "resnet_lidar", "weights")
        net.load_state_dict(torch.load(os.path.join(root, "resnet_lidar_v1_dropout_ep17.pt")))
        self.net = net.cuda()

        # Inertia
        self.stuck_detector = 0
        self.forced_move = 0




    def _init(self):

        """Initialized route planner"""

        self._route_planner = utils.RoutePlanner(self.config.route_planner_min_distance, self.config.route_planner_max_distance)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True

    def _get_position(self, tick_data):

        """converts gps position to route planner position
        Args:
            tick_data: processed input_data
        Returns:
            gps position
        """

        gps = tick_data['gps']
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    def sensors(self):

        """defines sensor suite for agent"""

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
                        },
                    {
                        'type': 'sensor.lidar.ray_cast',
                        'x': self.lidar_pos[0], 'y': self.lidar_pos[1], 'z': self.lidar_pos[2],
                        'roll': self.config.lidar_rot[0], 'pitch': self.config.lidar_rot[1], 'yaw': self.config.lidar_rot[2],
                        'id': 'lidar'
                        }
                    ]


        return sensors

    def tick(self, input_data):

        """Processes data to trainingsdata format

        Args:
            input_data: Sensor data retrieved from carla
        Returns:
            A dict mapping sensors to the respective processed data
        """

        # IMAGE PROCESSING
        rgb = []
        for pos in ['left', 'front', 'right']:
            rgb_cam = 'rgb_' + pos
            rgb_pos = cv2.cvtColor(input_data[rgb_cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            rgb_pos = utils.scale_crop(Image.fromarray(rgb_pos), self.config.scale, self.config.img_width, self.config.img_width, self.config.img_resolution[0], self.config.img_resolution[0])
            rgb.append(rgb_pos)
        rgb = np.concatenate(rgb, axis=1)

        # NAVIGATION
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']

        #print("speed"+str(speed))

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
        next_wp, next_cmd = waypoint_route[0]

        print(str(next_cmd))
        print("\n")

        result['next_command'] = next_cmd.value

        theta = compass + np.pi/2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
            ])

        local_command_point = np.array([next_wp[0]-denoised_pos[0], next_wp[1]-denoised_pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result['target_point'] = tuple(local_command_point)

        lidar = input_data['lidar'][1]#[:, :3]
        result['lidar'] = lidar

        return result



    @torch.inference_mode() # Faster version of torch_no_grad
    def run_step(self, input_data, timestamp):

        """Runs a decision making step of the agent.

        Also performs preprocessing steps necessary for being fed into the torch network like normalization, dimensionalitys etc.

        Args:
            input_data: Sensor data retrieved from carla
            timestamp: current time

        Returns:
            A dict mapping sensors to the respective processed data
        """

        self.step += 1

        if not self.initialized:
            self._init()
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            self.control = control        

        # Need to run this every step for GPS denoising
        tick_data = self.tick(input_data)

        # INERTIA
        is_stuck = False
        if(self.stuck_detector > self.config.stuck_threshold and self.forced_move < self.config.creep_duration):
            print("Detected agent being stuck. Move for frame: ", self.forced_move)
            is_stuck = True
            self.forced_move += 1
        print(self.stuck_detector)

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

        # LIDAR
        lidar = transform_lidar_bev(tick_data["lidar"])
        lidar = torch.tensor(lidar).float()
        lidar = torch.unsqueeze(lidar, 0)
        lidar = lidar.repeat((3, 1, 1)) # ResNet requires 3 channels, create 3 channels by copying
        lidar = torch.unsqueeze(lidar.to(device),0)
        lidar = preprocessing["lidar_bev"](lidar)

        #### FORWARD PASS,
        img_norm = img_norm.to(device)
        cmd_one_hot = torch.unsqueeze(cmd_one_hot.to(device),0)
        spd_norm = torch.unsqueeze(torch.unsqueeze(spd_norm.to(device), 0),0)

        with torch.no_grad():
            self.net.eval()
            outputs_ = self.net(img_norm,lidar, cmd_one_hot, spd_norm)
        brake, steer, throttle = outputs_

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

    def scale_crop(self, image, scale=1, start_x=0, crop_x=None, start_y=0, crop_y=None):

        """Scales and crops image to same format as Transfuser trainingdata

        Args:
            image: image to be scaled and cropped
            scale: index of route which is run from leaderboard
            start_x: pixel x from where to start cropping
            crop_x: how much to crop starting from x
            start_y: pixel y from where to start cropping
            crop_y: how much to crop starting from y
        """

        (width, height) = (image.width // scale, image.height // scale)
        if scale != 1:
            image = image.resize((width, height))
        if crop_x is None:
            crop_x = width
        if crop_y is None:
            crop_y = height
            
        image = np.asarray(image)
        cropped_image = image[start_y:start_y+crop_y, start_x:start_x+crop_x]

        return cropped_image

    def destroy(self):
        del self.net

