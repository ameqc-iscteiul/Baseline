#import math
from typing import List, Optional
import numpy as np
from mujoco import MjModel, MjData
from revolve2.serialization import StaticData, Serializable
import abrain
import utils
from abrain import Point, plotly_render
from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import Brain, Body, ActiveHinge
from .vision import OpenGLVision
from robot.genome import RVGenome
from simulation.runner import DefaultActorControl


# ==============================================================================
# Basic control data: (built-in) sensors
# ==============================================================================

class SensorControlData(DefaultActorControl):
    def __init__(self, mj_model: MjModel, mj_data: MjData):
        DefaultActorControl.__init__(self, mj_model, mj_data)
        self.sensors = mj_data.sensordata
        self.model, self.data = mj_model, mj_data


# ==============================================================================
# ANN (abrain) controller (also handles the camera)
# ==============================================================================


class ANNControl:
    class Controller(ActorController):
        def __init__(self, genome: abrain.Genome, inputs: List[Point], outputs: List[Point]):
            self.brain = abrain.ANN.build(inputs, outputs, genome)
            self.i_buffer, self.o_buffer = self.brain.buffers()
            self.vision: Optional[OpenGLVision] = None
            self._step = 0
            self.lum_input_list=[]

            self.current_vision_input=[]

        def get_ANN(self):
            return self.brain

        def get_dof_targets(self) -> List[float]:
            return [self.o_buffer[i] for i in range(len(self.o_buffer))]

        def get_robot_vision(self):
            return self.lum_input_list

        def get_luminosity_input(self,img):
            lum_input=[]
            #First Method, Averaging RGB values
            
            luminosity_matrix = np.average(img, axis=2)
            lum_input = [x/255  for x in luminosity_matrix.flat]
            '''

            #Second, Using the Luminosity Method 
            #(our eyes percieve each RGB component with a dif weight)
            for line in img:
                for rgb in line:
                    R, G, B = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
                    #try without rounding
                    #lum_input.append(0.2126 * R + 0.7152 * G + 0.0722 * B)
                    lum_input.append(round(0.213 * R + 0.715 * G + 0.072 * B, 3))
            '''
            return lum_input
        
        def step(self, dt: float, data: 'ControlData') -> None:
            input=[]
            #1-Vision Input
            img = self.vision.process(data.model, data.data)
            lum_input = self.get_luminosity_input(img)
            input.extend(lum_input)

            '''#2-Sin Input
            freq=0.01
            sin_input = math.sin(2 * math.pi * freq) * self._step
            input.append(sin_input)'''
            
            #3-Hinge Input
            hinge_input = [pos for pos in data.sensors]
            input.extend(hinge_input)

            self.i_buffer[:] = input
            self._step += 1
            self.brain.__call__(self.i_buffer, self.o_buffer)
            self.lum_input_list.append(lum_input)


        
        @classmethod
        def deserialize(cls, data: StaticData) -> Serializable:
            raise NotImplementedError

        def serialize(self) -> StaticData:
            raise NotImplementedError


    class Brain(Brain):
        def __init__(self, brain_dna: RVGenome):
            self.brain_dna = brain_dna
      
        def make_controller(self, body: Body, dof_ids: List[int]) -> ActorController:
            
            inputs, outputs = [], []
            active_hinges_unsorted = body.find_active_hinges()
            active_hinge_map = {
                active_hinge.id: active_hinge for active_hinge in active_hinges_unsorted
            }
            active_hinges = [active_hinge_map[id] for id in dof_ids]
            w, h = self.brain_dna.vision
               
            inputs=utils.distribute_points(-1,w,h)
            #inputs.append(Point(0,-0.8,0))
            hinge_input=utils.distribute_points(-0.9,len(active_hinges),1)
            inputs.extend(hinge_input)

            outputs=utils.distribute_points(1,len(active_hinges),1)

            return ANNControl.Controller(self.brain_dna.brain, inputs, outputs)

            
            
            