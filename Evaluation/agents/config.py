import os

# TRANSFUSER CONFIGURATION OF SENSORS ETC

class RGB_Config:
    scale = 1
    img_resolution = (160, 704)  # image pre-processing in H, W
    img_width = 320  # important this should be consistent with scale, e.g. scale = 1, img_width 320, scale=2, image_width 640

    camera_pos = [1.3, 0.0, 2.3] #x, y, z mounting position of the camera
    camera_width = 960 # Camera width in pixel
    camera_height = 480 # Camera height in pixel
    camera_fov = 120 #Camera FOV in degree
    camera_rot_0 = [0.0, 0.0, 0.0] # Roll Pitch Yaw of camera 0 in degree
    camera_rot_1 = [0.0, 0.0, -60.0] # Roll Pitch Yaw of camera 1 in degree
    camera_rot_2 = [0.0, 0.0, 60.0] # Roll Pitch Yaw of camera 2 in degree

    # Carla Simulation Settings
    gps_buffer_max_len = 100 # Number of past gps measurements that we track.
    carla_frame_rate = 1.0 / 20.0 # CARLA frame rate in milliseconds
    carla_fps = 20 # Simulator Frames per second
    iou_treshold_nms = 0.2  # Iou threshold used for Non Maximum suppression on the Bounding Box predictions for the ensembles
    steer_damping = 0.5 # Damping factor by which the steering will be multiplied when braking
    route_planner_min_distance = 7.5
    route_planner_max_distance = 50.0
    action_repeat = 2 # Number of times we repeat the networks action. It's 2 because the LiDAR operates at half the frame rate of the simulation

    # Inertia Settings
    stuck_threshold = 2000 /action_repeat # Number of frames after which the creep controller starts triggering. Inertia param old : 2000
    creep_duration = 30 / action_repeat # Number of frames we will creep forward, DIFFERENT FROM LIDAR

class Lidar_Config:
    scale = 1
    img_resolution = (160, 704)  # image pre-processing in H, W
    img_width = 320  # important this should be consistent with scale, e.g. scale = 1, img_width 320, scale=2, image_width 640

    camera_pos = [1.3, 0.0, 2.3]  # x, y, z mounting position of the camera
    camera_width = 960  # Camera width in pixel
    camera_height = 480  # Camera height in pixel
    camera_fov = 120  # Camera FOV in degree
    camera_rot_0 = [0.0, 0.0, 0.0]  # Roll Pitch Yaw of camera 0 in degree
    camera_rot_1 = [0.0, 0.0, -60.0]  # Roll Pitch Yaw of camera 1 in degree
    camera_rot_2 = [0.0, 0.0, 60.0]  # Roll Pitch Yaw of camera 2 in degree

    lidar_resolution_width  = 256 # Width of the LiDAR grid that the point cloud is voxelized into.
    lidar_resolution_height = 256 # Height of the LiDAR grid that the point cloud is voxelized into.
    pixels_per_meter = 8.0 # How many pixels make up 1 meter. 1 / pixels_per_meter = size of pixel in meters
    lidar_pos = [1.3,0.0,2.5] # x, y, z mounting position of the LiDAR
    lidar_rot = [0.0, 0.0, -90.0] # Roll Pitch Yaw of LiDAR in degree

    # Carla Simulation Settings
    gps_buffer_max_len = 100  # Number of past gps measurements that we track.
    carla_frame_rate = 1.0 / 20.0  # CARLA frame rate in milliseconds
    carla_fps = 20  # Simulator Frames per second
    iou_treshold_nms = 0.2  # Iou threshold used for Non Maximum suppression on the Bounding Box predictions for the ensembles
    steer_damping = 0.5  # Damping factor by which the steering will be multiplied when braking
    route_planner_min_distance = 7.5
    route_planner_max_distance = 50.0
    action_repeat = 2  # Number of times we repeat the networks action. It's 2 because the LiDAR operates at half the frame rate of the simulation

    # Inertia Settings
    stuck_threshold = 2000 / action_repeat  # Number of frames after which the creep controller starts triggering. Inertia param old : 2000
    creep_duration = 45 / action_repeat  # Number of frames we will creep forward # DIFFERENT FROM RGB, because it needs stronger push
