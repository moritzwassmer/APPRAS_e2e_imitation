#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an NPC agent to control the ego vehicle
"""

from __future__ import print_function

import carla
from agents.navigation.basic_agent import BasicAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

from config import GlobalConfig

def get_entry_point():
    return 'NpcAgent'

class NpcAgent(AutonomousAgent):

    """
    NPC autonomous agent to control the ego vehicle
    """

    _agent = None
    _route_assigned = False

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self.track = Track.SENSORS

        self.config = GlobalConfig(setting='eval')

        self._route_assigned = False
        self._agent = None

    def sensors(self):
        sensors = [
            {
                'type': 'sensor.camera.rgb',
                'x': self.config.camera_pos[0], 'y': self.config.camera_pos[1], 'z': self.config.camera_pos[2],
                'roll': self.config.camera_rot_0[0], 'pitch': self.config.camera_rot_0[1],
                'yaw': self.config.camera_rot_0[2],
                'width': self.config.camera_width, 'height': self.config.camera_height, 'fov': self.config.camera_fov,
                'id': 'rgb_front'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': self.config.camera_pos[0], 'y': self.config.camera_pos[1], 'z': self.config.camera_pos[2],
                'roll': self.config.camera_rot_1[0], 'pitch': self.config.camera_rot_1[1],
                'yaw': self.config.camera_rot_1[2],
                'width': self.config.camera_width, 'height': self.config.camera_height, 'fov': self.config.camera_fov,
                'id': 'rgb_left'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': self.config.camera_pos[0], 'y': self.config.camera_pos[1], 'z': self.config.camera_pos[2],
                'roll': self.config.camera_rot_2[0], 'pitch': self.config.camera_rot_2[1],
                'yaw': self.config.camera_rot_2[2],
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
                'x': self.config.lidar_pos[0], 'y': self.config.lidar_pos[1], 'z': self.config.lidar_pos[2],
                'roll': self.config.lidar_rot[0], 'pitch': self.config.lidar_rot[1], 'yaw': self.config.lidar_rot[2],
                'id': 'lidar'
            }#,
            #{
            #    'type': 'sensor.opendrive_map', # not used by transfuser
            #    'reading_frequency': 1,
            #    'id': 'OpenDRIVE'
            #}
        ]

        return sensors

    def run_step(self, input_data, timestamp):

        print("=====================>")
        for key, val in input_data.items():
            if hasattr(val[1], 'shape'):
                shape = val[1].shape
                print("[{} -- {:06d}] with shape {}, type {}".format(key, val[0], shape, type(val[1])))
            else:
                print("[{} -- {:06d}], type {}".format(key, val[0], type(val[0])))
        print("<=====================")

        """
        Execute one step of navigation.
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        if not self._agent:
            hero_actor = None
            for actor in CarlaDataProvider.get_world().get_actors():
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                    hero_actor = actor
                    break
            if hero_actor:
                self._agent = BasicAgent(hero_actor) # agent which is able to drive through waypoints

            return control

        if not self._route_assigned:
            if self._global_plan:
                plan = []

                prev = None
                for transform, _ in self._global_plan_world_coord:
                    wp = CarlaDataProvider.get_map().get_waypoint(transform.location) #current location
                    if  prev:
                        route_segment = self._agent._trace_route(prev, wp)
                        plan.extend(route_segment)

                    prev = wp
                #print(plan)

                #loc = plan[-1][0].transform.location
                #self._agent.set_destination([loc.x, loc.y, loc.z])
                self._agent._local_planner.set_global_plan(plan)  # pylint: disable=protected-access
                self._route_assigned = True

        else:
            control = self._agent.run_step()

        return control
