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
import cv2
import glfw
import mujoco
import numpy as np
from mujoco import MjModel, MjData
from mujoco_viewer import mujoco_viewer
from pyrr import Vector3, Quaternion
from robot.vision import OpenGLVision
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


class CallbackType(Enum):
    VIDEO_FRAME_CAPTURED = auto()

RunnerCallback = Callable[[float, MjModel, MjData], None]
MovieCallback = Callable[[np.ndarray], None]
RunnerCallbacks = Dict[CallbackType, Union[RunnerCallback, MovieCallback]]

@dataclass
class RunnerOptions:
    #Scenario options
    robot_position = [-2, 0, 0]
    target_position = [2, 0, 0]
    target_size : float = 0.5

    level:int=0
    return_ann=False
    
    @dataclass
    class View:
        start_paused: bool = False
        speed: float = 1.0
        auto_quit: bool = True
        cam_id: Optional[int] = 0
        settings_restore: bool = True
        settings_save: bool = True
        mark_start: bool = True
    view: Optional[View] = None

    @dataclass
    class Record:
        video_file_path: Path
        width: int = 640
        height: int = 480
        fps: int = 24
    record: Optional[Record] = None

    save_folder: Optional[Path] = None

class DefaultActorControl(ActorControl):
    def __init__(self, mj_model: MjModel, mj_data: MjData):
        ActorControl.__init__(self)

class DefaultEnvironmentActorController(EnvironmentActorController):
    def control(self, dt: float, actor_control: DefaultActorControl) -> None:
        self.actor_controller.step(dt, actor_control)
        actor_control.set_dof_targets(0, self.actor_controller.get_dof_targets())

class Runner(LocalRunner):
    environmentActorController_t = DefaultEnvironmentActorController
    actorController_t = DefaultActorControl

    def __init__(self, robot: ModularRobot, options: RunnerOptions,
                 env_seeder: Callable[[mjcf.RootElement, RunnerOptions], None],
                 callbacks: Optional[RunnerCallbacks] = None):

        LocalRunner.__init__(self)

        self.options = options
        #print("Level:", options.level)
        self.bonus = 0

        self.target_position=options.target_position
        self.target_size=options.target_size
        #target_area_side = self.target_size + .07
        # Calculate the minimum and maximum boundaries of the target area
        '''self.target_area_intervals = [round(self.target_position[0] - target_area_side,3),
                                    round(self.target_position[0] + target_area_side,3),
                                    round(self.target_position[1] - target_area_side,3),
                                    round(self.target_position[1] + target_area_side,3)]'''

        self.callbacks = {}
        if callbacks is not None:
            self.callbacks.update(callbacks)
            
        actor, controller = robot.make_actor_and_controller()
        bounding_box = actor.calc_aabb()
        env = Environment(self.environmentActorController_t(controller))

        
        robot_position = Vector3(options.robot_position)
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
        self.video = None
        if (self.options.view is None and self.options.record is None):
            self.headless = True
        else: 
            self.headless = False
            if self.options.record is not None:
                self.prepare_video(self.options)
            elif self.options.view is not None:
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

    def prepare_video(self, options):
        self.video = namedtuple('Video', ['step', 'writer', 'last'])
        self.video.step = 1 / options.record.fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        size = (options.record.width, options.record.height)
        self.video.writer = cv2.VideoWriter(
            str(options.record.video_file_path),
            fourcc=fourcc,
            fps=options.record.fps,
            frameSize=size
        )
        self.video.last = 0.0
        self.viewer = OpenGLVision(model=self.model, shape=size)
        self.viewer.cam.fixedcamid = 0
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED        

    @staticmethod
    def _config_file():
        return os.path.join(os.path.expanduser('~'),
                            '.config/revolve/viewer.ini')
    def update_view(self, time):
        if self.video is None:
            self.viewer.render()

        # capture video frame if it's time
        if self.video is not None and time >= self.video.last + self.video.step:
            self.video.last = int(time / self.video.step) * self.video.step

            frame = cv2.cvtColor(
                np.flipud(self.viewer.process(self.model, self.data)),
                cv2.COLOR_RGBA2BGR)
            
            flag = CallbackType.VIDEO_FRAME_CAPTURED
            if flag in self.callbacks:
                self.callbacks[flag](frame)

            self.video.writer.write(frame)

    def close_view(self):
        if self.options.view is not None and self.options.view.auto_quit:
            self.viewer.close()
        if self.options.record is not None:
            self.video.writer.release()

    def run(self):

        last_sample_time = 0.0
        last_control_time = 0.0
        control_step = 1 / Config.control_frequency
        sample_step = 1 / Config.sampling_frequency
        results = EnvironmentResults([])

        # sample initial state
        results.environment_states.append(
            EnvironmentState(0.0, self.get_actor_state(0))
        )
        
        #RUN LOOP
        full_gaze=0
        target_reached=False
        while ((time := self.data.time) < Config.simulation_time ) and target_reached is False:
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

            # sample state if it is time
            if time >= last_sample_time + sample_step:
                last_sample_time = int(time / sample_step) * sample_step
                results.environment_states.append(EnvironmentState(time, self.get_actor_state(0)))
            
            n=3
            vision = self.controller.actor_controller.get_robot_vision()
            #if the current gaze was full, check if the last n gazes were also full
            if len(vision) > n and sum(vision[-n]) == len(vision[0]):
                if sum([value for view in vision[-n:] for value in view])==(len(vision[0])*n):
                    self.bonus = Config.simulation_time - time
                    target_reached=True
                        #print(f" !!!  Robot gazed Target {n} consecutive times  !!!")



            '''if sum(self.controller.actor_controller.get_current_vision_input())==9:
                print( "Gaze:", full_gaze," - VISION: ",self.controller.actor_controller.get_current_vision_input())
                full_gaze+=1
                if full_gaze > 10:
                    print("Robot gazed Target 2s!")
                    self.bonus = Config.simulation_time - time
                    target_reached=True
            else:
                full_gaze=0'''
            #If it gazes at target for 2 seconds
            # 10 controls/s so if in 10 controls the robot is seeing white,
            # it means he was looking for 1 s
            

            if not self.headless:
                self.update_view(time)
        if not self.headless:
            self.close_view()
        return results, self.bonus
    
    def get_actor_state(self, robot_index):
        return self._get_actor_state(robot_index, self.data, self.model)

        


    