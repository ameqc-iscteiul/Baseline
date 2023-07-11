import math
from dataclasses import dataclass
from functools import partial
import os
from pathlib import Path
from typing import Callable, Dict, Optional, Union
from configparser import ConfigParser
from ast import literal_eval
from collections import namedtuple
from enum import Enum, auto, Flag

import glfw
import mujoco
import numpy as np
from mujoco import MjModel, MjData
from mujoco_viewer import mujoco_viewer
from pyrr import Vector3, Quaternion

from revolve2.core.modular_robot import ModularRobot
from revolve2.core.physics.environment_actor_controller import EnvironmentActorController
from revolve2.runners.mujoco import LocalRunner

from revolve2.runners.mujoco._local_runner import mjcf
from revolve2.core.physics.running import (
    ActorControl,
    Environment,
    EnvironmentResults,
    EnvironmentState,
    PosedActor
)

from simulation.config import Config
from simulation.my_runner import RunnerOptions

class CallbackType(Enum):
    VIDEO_FRAME_CAPTURED = auto()

RunnerCallback = Callable[[float, MjModel, MjData], None]
MovieCallback = Callable[[np.ndarray], None]
RunnerCallbacks = Dict[CallbackType, Union[RunnerCallback, MovieCallback]]


class DefaultActorControl(ActorControl):
    def __init__(self, mj_model: MjModel, mj_data: MjData):
        ActorControl.__init__(self)

class DefaultEnvironmentActorController(EnvironmentActorController):
    def control(self, dt: float, actor_control: DefaultActorControl) -> None:
        self.actor_controller.step(dt, actor_control)
        actor_control.set_dof_targets(0, self.actor_controller.get_dof_targets())

class ReRunner(LocalRunner):
    environmentActorController_t = DefaultEnvironmentActorController
    actorController_t = DefaultActorControl

    def __init__(self, robot: ModularRobot, options: RunnerOptions,
                 env_seeder: Callable[[mjcf.RootElement, RunnerOptions], None],
                 callbacks: Optional[RunnerCallbacks] = None,
                 robot_position: Optional[Vector3] = None,
                 target_position: Optional[Vector3] = None,
                 target_size: Optional[float] = None):

        LocalRunner.__init__(self)

        self.options = options
        self.bonus = 0
        self.target_position=target_position
        self.target_size=target_size
        self.callbacks = {}
        if callbacks is not None:
            self.callbacks.update(callbacks)
            
        actor, controller = robot.make_actor_and_controller()
        bounding_box = actor.calc_aabb()
        env = Environment(self.environmentActorController_t(controller))

        if robot_position is None:
            robot_position = Vector3([0, 0, 0])
        else:
            robot_position = Vector3(robot_position)
        robot_position = Vector3([
            robot_position.x, robot_position.y,
            robot_position.z + bounding_box.size.z / 2.0 - bounding_box.offset.z
        ])
        env.actors.append(
            PosedActor(
                actor,
                robot_position,
                Quaternion(),
                [0.0 for _ in controller.get_dof_targets()],
            )
        )
        self.controller = env.controller

        self.model = mujoco.MjModel.from_xml_string(
            LocalRunner._make_mjcf(env,
                                   partial(env_seeder, options=options)))

        self.data = mujoco.MjData(self.model)

        initial_targets = [
            dof_state
            for posed_actor in env.actors
            for dof_state in posed_actor.dof_states
        ]
        LocalRunner._set_dof_targets(self.data, initial_targets)

        self.headless = False

        glfw_window_hint = {
            Config.OpenGLLib.OSMESA.name: glfw.OSMESA_CONTEXT_API,
            Config.OpenGLLib.EGL.name: glfw.EGL_CONTEXT_API
        }
        if (ogl := Config.opengl_lib.upper()) in glfw_window_hint:
            glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw_window_hint[ogl])
            print("Requested:", ogl)

        viewer = mujoco_viewer.MujocoViewer(
                self.model,
                self.data,
            )
        viewer._render_every_frame = False  # Private but functionality is not exposed and for now it breaks nothing.
        viewer._paused = False
        self.viewer = viewer
        self.viewer.cam.lookat = self.get_actor_state(0).position

            
    @staticmethod
    def _config_file():
        return os.path.join(os.path.expanduser('~'),
                            '.config/revolve/viewer.ini')

    def close_view(self):
        if self.options.view is not None and self.options.view.auto_quit:
            self.viewer.close()
    

    def run(self):
        last_sample_time = 0.0
        results = EnvironmentResults([])

        # sample initial state
        results.environment_states.append(
            EnvironmentState(0.0, self.get_actor_state(0))
        )

        last_control_time = 0.0

        control_step = 1 / Config.control_frequency

        target_area_side = self.target_size + .04
        # Calculate the minimum and maximum boundaries of the target area
        target_area_x_interval = [round(self.target_position[0] - target_area_side,2),
                                round(self.target_position[0] + target_area_side,2)]
        target_area_y_interval = [round(self.target_position[1] - target_area_side,2),
                                round(self.target_position[1] + target_area_side,2)]
        #print("target_area_x_interval",target_area_x_interval,"target_area_y_interval", target_area_y_interval)
        
        #RUN LOOP
        while (time := self.data.time) < Config.simulation_time:
            # do control if it is time
            is_control_step = (time >= last_control_time + control_step)

            if is_control_step:
                is_control_step = True
                last_control_time = \
                    math.floor(time / control_step) * control_step
                control_user = self.actorController_t(self.model, self.data)
                self.controller.control(control_step, control_user)
                actor_targets = control_user._dof_targets
                actor_targets.sort(key=lambda t: t[0])
                targets = [
                    target
                    for actor_target in actor_targets
                    for target in actor_target[1]
                ]
                LocalRunner._set_dof_targets(self.data, targets)

            # step simulation
            mujoco.mj_step(self.model, self.data)
            # Check if the robot's position is within the target area
            robot_position = self.get_actor_state(0).position
            if target_area_x_interval[0] <= round(robot_position[0],2) <= target_area_x_interval[1] and  target_area_y_interval[0] <= round(robot_position[1],2) <= target_area_y_interval[1]: 
                print("Robot reached Target!")
                self.bonus = time/Config.simulation_time
                break

            # sample state if it is time
            if time >= last_sample_time + Config.sampling_frequency:
                last_sample_time = int(time / Config.sampling_frequency) * Config.sampling_frequency
                results.environment_states.append(EnvironmentState(time, self.get_actor_state(0)))
            
            self.viewer.render()
        
        self.close_view()

        # sample one final time
        results.environment_states.append(
            EnvironmentState(time, self.get_actor_state(0))
        )
        return results, self.bonus
    
    def get_actor_state(self, robot_index):
        return self._get_actor_state(robot_index, self.data, self.model)

        


    