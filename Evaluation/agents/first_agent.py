#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a dummy agent to control the ego vehicle
"""

from __future__ import print_function

import carla

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track


import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense # TODO
import numpy as np
from pickle import load
from PIL import Image

def get_entry_point():
    return 'DummyAgent'

def conv_steer(pred):
    if pred > 1:
        return 1
    if pred < -1:
        return -1
    return pred

def conv_throttle_brake(pred):
    if pred > 1:
        return 1
    if pred < 0:
        return 0
    return pred

class DummyAgent(AutonomousAgent):

    """
    Dummy autonomous agent to control the ego vehicle
    """

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self.track = Track.SENSORS

        #print("CALLLEDDDDDDDD")

        #self.model = Sequential()
        #self.model.add(Dense(50, input_shape=(460800,), activation='relu'))
        #self.model.add(Dense(3, activation='linear'))

        #self.model.summary()
        #self.model.compile(loss='mse', optimizer='adam', metrics=['mse'])

        self.model = keras.models.load_model('C:/Users/morit/OneDrive/UNI/Master/WS22/APP-RAS/Programming/myfolder')
        self.rgb_scaler = load(open('C:/Users/morit/OneDrive/UNI/Master/WS22/APP-RAS/Programming/myfolder/rgb_scaler.pkl', 'rb'))

    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]
        """


        #FROM config.py of sumbmission agent
        camera_pos = [1.3, 0.0, 2.3]  # x, y, z mounting position of the camera
        camera_width = 960  # Camera width in pixel
        camera_height = 160#480  # Camera height in pixel
        camera_fov = 120  # Camera FOV in degree
        camera_rot_0 = [0.0, 0.0, 0.0]  # Roll Pitch Yaw of camera 0 in degree

        sensors = [
            {
                'type': 'sensor.camera.rgb',
                'x': camera_pos[0], 'y': camera_pos[1], 'z': camera_pos[2],
                'roll': camera_rot_0[0], 'pitch': camera_rot_0[1],
                'yaw': camera_rot_0[2],
                'width': camera_width, 'height': camera_height, 'fov': camera_fov,
                'id': 'rgb_front'
            },

        ]
        """
        sensors = [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 800, 'height': 600, 'fov': 100, 'id': 'Center'}
        ]
        """

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """

        #print("=====================>")
        for key, val in input_data.items():

            if key == "rgb_front":
                #print(np.shape(val[1]))
                X = self.preprocess_image(val[1])
                #print(X)
                Preds = self.model.predict(X)  # measurements["steer"],measurements["throttle"],measurements["brake"]


                Preds = Preds.flatten()
                #print(np.shape(Preds))

                # RETURN CONTROL
                control = carla.VehicleControl()
                #print(conv_steer(Preds[0]))
                print(conv_steer(Preds[0]))
                control.steer = float(conv_steer(Preds[0]))#Preds[0]
                print(conv_throttle_brake(Preds[1]))
                control.throttle = float(conv_throttle_brake(Preds[1]))#Preds[1]
                print(conv_throttle_brake(Preds[2]))
                control.brake = float(conv_throttle_brake(Preds[2]))#Preds[2]
                control.hand_brake = False

                return control

            if hasattr(val[1], 'shape'):
                shape = val[1].shape
                #print("[{} -- {:06d}] with shape {}".format(key, val[0], shape))
            else:
                a = 1
                #print("[{} -- {:06d}] ".format(key, val[0]))
        #print("<=====================")


        # RETURN CONTROL default control
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        return control



    def preprocess_image(self, image):

        PIL_image = Image.fromarray(np.uint8(image)).convert('RGB')
        image = np.asarray(PIL_image)[:, :, :3]

        #print(image)

        scaled_flattened_image = self.rgb_scaler.transform(image.flatten().reshape(1, -1))
        #print(scaled_flattened_image)
        #print(np.shape(scaled_flattened_image))

        """
        #TODO Preprocess and sclae with OLD SCALER
        test = load(open('C:/Users/morit/OneDrive/UNI/Master/WS22/APP-RAS/Programming/myfolder/rgb_scaler.pkl', 'rb'))
        """

        return scaled_flattened_image
