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
            image_data = self.vision.process(data.model, data.data)
            lum_input = self.get_luminosity_input(image_data)
            #print("lum_input:   ", lum_input)
            self.lum_input_list.append(lum_input)
            input.extend(lum_input)
            #2-Hinge Input
            hinge_input = [pos for pos in data.sensors]
            input.extend(hinge_input)
            self.i_buffer[:] = input
            self._step += 1
            self.brain.__call__(self.i_buffer, self.o_buffer)


        
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
            #1-Vision Input
            w, h = self.brain_dna.get_vision()
            inputs=utils.distribute_points(-1, w, h)
            #2-Hinge Input and Output
            parsed_coords = body.to_tree_coordinates()
            bounds = np.zeros((2, 3), dtype=int)
            np.quantile([c[1].tolist() for c in parsed_coords], [0, 1], axis=0, out=bounds)

            if bounds[0][2] != bounds[1][2]:
                raise NotImplementedError("Can only handle planar robots (with z=0 for all modules)")
            x_min, x_max = bounds[0][0], bounds[1][0]
            xrange = max(x_max-x_min, 1)
            y_min, y_max = bounds[0][1], bounds[1][1]
            yrange = max(y_max-y_min, 1)

            hinges_map = {
                c[0].id: (
                    2 * (c[1].x - x_min) / xrange - 1 if xrange > 1 else 0,
                    2 * (c[1].y - y_min) / yrange - 1 if yrange > 1 else 0)
                for c in parsed_coords if isinstance(c[0], ActiveHinge)
            }
            for i, did in enumerate(dof_ids):
                    p = hinges_map[did]
                    ip = Point(p[1], -0.6, p[0])
                    inputs.append(ip)
                    op = Point(p[1], 1, p[0])
                    outputs.append(op)

            '''
            Old Controller
            #make controller:
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
            '''


            '''#1-Vision Input
            image_data = self.vision.process(data.model, data.data)
            #print("img",image_data)
            
            # Calculate the midpoint index along the vertical centerline
            midpoint_index = image_data.shape[1] // 2
            # Split the image into left and right sub-images
            left_subimg = image_data[:, :midpoint_index]
            right_subimg = image_data[:, midpoint_index:]

            #print('left_half',self.get_luminosity_input(left_subimg))
            #print('right_half',self.get_luminosity_input(right_subimg))
            left_lum=self.get_luminosity_input(left_subimg)
            right_lum=self.get_luminosity_input(right_subimg)
            #print("right_lum",right_lum)
            
            # Apply the contrast filter to each pixel in the array
            # "make darker go darker" (dark corresponds to lower then bound)
            left_lum = np.clip(left_lum, 0.0, 1.0)
            filtered_left_lum = list(np.vectorize(lambda pixel: utils.contrast_filter(pixel, contrast_factor=0.5, bound_value=0.5))(left_lum))
            right_lum = np.clip(right_lum, 0.0, 1.0)
            filtered_right_lum = list(np.vectorize(lambda pixel: utils.contrast_filter(pixel, contrast_factor=0.5, bound_value=0.5))(right_lum))

            left_score= sum(filtered_left_lum)/len(filtered_left_lum)
            right_score= sum(filtered_right_lum)/len(filtered_right_lum)

            lum_input = [left_score, right_score]
            '''

            return ANNControl.Controller(self.brain_dna.brain, inputs, outputs)
        
        

        

            
            
            