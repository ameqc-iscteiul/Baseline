import math
from typing import Optional, List, Dict

import cv2
import numpy as np
from pyrr import Vector3
from revolve2.core.modular_robot import Body, Brick, ActiveHinge, ModularRobot, Module
from simulation.config import Config
from robot.control import ANNControl, SensorControlData
from robot.genome import RVGenome, VisionData
from simulation.runner import Runner,RunnerOptions
# ==============================================================================
# Robot
# ==============================================================================

Runner.actorController_t = SensorControlData

def GECKO()-> Body:
    def add_arms(m: Module):
        m.left = ActiveHinge(math.pi / 2.0)
        m.left.attachment = ActiveHinge(math.pi / 2.0)
        m.left.attachment.attachment = Brick(0.0)
        m.right = ActiveHinge(math.pi / 2.0)
        m.right.attachment = ActiveHinge(math.pi / 2.0)
        m.right.attachment.attachment = Brick(0.0)
    body = Body()
    body.core.front = Brick(0.0)
    body.core.back = Brick(0.0)
    for side in ['front', 'back']:
        brick = Brick(0.0)
        setattr(body.core, side, brick)
        add_arms(brick)
    body.finalize()
    return body

def default():
    body = Body()
    body.core.left = ActiveHinge(math.pi / 2.0)
    body.core.left.attachment = ActiveHinge(math.pi / 2.0)
    body.core.left.attachment.attachment = Brick(0.0)
    body.core.right = ActiveHinge(math.pi / 2.0)
    body.core.right.attachment = ActiveHinge(math.pi / 2.0)
    body.core.right.attachment.attachment = Brick(0.0)
    body.finalize()
    return body


def build_robot(brain_dna: RVGenome, vision_w, vision_h, type):
    brain_dna.set_vision(vision_w,vision_h)
    if type==0:
        robot = default()
    elif type==1:
        robot = GECKO()
    else:
        print("Wrong Body Type")
    return ModularRobot(robot, ANNControl.Brain(brain_dna))

# ==============================================================================
# Scenario
# ==============================================================================

class Scenario:

    target_pos=""
    target_size=0.0

    def __init__(self, run_id: Optional[int] = None):
        #self.runner = runner
        self.id = run_id
        
    @classmethod
    def insert_target(cls,target_pos,target_size):
        cls.target_pos=' '.join([str(x) for x in target_pos])
        cls.target_size=target_size
        
    def assign_runner(self, runner:Runner):
        self.runner = runner
    

    # ==========================================================================

    @staticmethod
    def initial_position():
        return [-2, 0, 0]

    # ==========================================================================

    @staticmethod
    def amend(xml, options: RunnerOptions):
        robots = [r for r in xml.worldbody.body]

        xml.visual.map.znear = ".001"

        # Reference to the ground
        ground = next(item for item in xml.worldbody.geom if item.name == "ground")
        g_size = Config.ground_size
        gh_size = .5 * g_size

        # Remove default ground
        ground.remove()

        # Add texture and material for the ground
        xml.asset.add('texture', name="grid", type="2d", builtin="checker",
                      width="512", height="512",
                      rgb1="0.05 0.1 0.15", rgb2="0.1 0.15 0.2")

        xml.asset.add('material', name="grid", texture="grid",
                      texrepeat="1 1", texuniform="true", reflectance="0")
        xml.worldbody.add('geom', name="floor",
                          size=[gh_size, gh_size, .05],
                          type="plane", material="grid", condim=3)        
        #Borders
        gh_width = .025
        for i, x, y in [(0, 0, 1), (1, 1, 0), (2, 0, -1), (3, -1, 0)]:
            b_height = g_size / 100
            xml.worldbody.add('geom', name=f"border#{i}",
                              pos=[x * (gh_size + gh_width),
                                   y * (gh_size + gh_width), b_height],
                              rgba=[0.01, 0.01, 0.01, 0.001],
                              euler=[0, 0, i * math.pi / 2],
                              type="box", size=[gh_size, gh_width, b_height])
        #Target
        xml.worldbody.add(
            "geom",
            type="box",
            pos=Scenario.target_pos,
            size=f"{str(Scenario.target_size)} {str(Scenario.target_size)} 2",
            rgba="255 255 255 1"
        )


        def create_grid_positions(center_position, num_lines, elements_per_line, distance):
            positions = []

            line_offset = (elements_per_line - 1) / 2
            element_offset = (num_lines - 1) / 2

            for i in range(elements_per_line):
                y = center_position[1] - line_offset + i
                for j in range(num_lines):
                    x = center_position[0] - element_offset + j
                    position = [x * distance, y * distance, center_position[2]]
                    positions.append(position)

            return positions

        def create_obstacle_grid(xml,center_position ,num_lines, elements_per_line, size_of_side, distance):
            positions=create_grid_positions(center_position, num_lines, elements_per_line, distance)
            for p in positions:
                xml.worldbody.add(
                    "geom",
                    type="box",
                    pos=f"{p[0]} {p[1]} 0",
                    size=f"{size_of_side} {size_of_side} {size_of_side}",
                    rgba="0.5 0 0 1"
                )
            return xml
        
        def create_Ramp(xml, center_position, num_lines, elements_per_line, size_of_side, distance, height_step):
            positions = create_grid_positions(center_position, num_lines, elements_per_line, distance)
    
            max_distance_x = max(abs(center_position[0] - p[0]) for p in positions)
            
            for p in positions:
                distance_x = abs(center_position[0] - p[0])
                height = size_of_side + (max_distance_x - distance_x) * height_step
                
                xml.worldbody.add(
                    "geom",
                    type="box",
                    pos=f"{p[0]} {p[1]} 0",
                    size=f"{size_of_side} {size_of_side} {height}",
                    rgba="1 0 0 1"
                )
            
            return xml
        

        def insert_rectangle(xml, position, base_length, base_width, base_height):
            xml.worldbody.add(
                        "geom",
                        type="box",
                        pos=position,
                        size=f"{base_length} {base_width} {base_height}",
                        rgba="0.2 0.2 0 1"
                    )
            return xml
        def generate_mountain(xml, position, base_length, base_width, base_height, levels):
            for l in range(levels+1):
                xml=insert_rectangle(xml, position, base_length, base_width, base_height)
                base_length-=base_length*.20
                base_width-=base_width*.20
                next_z = float(position.split()[-1])+base_height
                #print(position.split()[0] + position.split()[1])
                #print(next_z)
                position = f'{position.split()[0]} {position.split()[1]} {str(next_z)}'

                #print(position)

            return xml
        
        def generate_deception(xml, x=1, side=0.03, height=0.25, space=0.09, level=2):
            
            xml = insert_rectangle(xml, f'{x} 0 0', side, side, height)
            y=space
            for _ in range(level):
                x-=space
                xml = insert_rectangle(xml, f'{x} {y} 0', side, side, height)
                xml = insert_rectangle(xml, f'{x} {-y} 0', side, side, height)
                y+=space
        
            return xml
        
        if options.level==1:
            xml = create_obstacle_grid(xml, [-2.5,0,0], 1, 8, 0.02, 0.3)
        if options.level==2:
            xml = create_obstacle_grid(xml, [-2.5,0,0], 1, 8, 0.02, 0.3)
            xml = create_obstacle_grid(xml, [-2.5,0,0], 2, 9, 0.02, 0.3) 
        if options.level==3:
            xml = create_obstacle_grid(xml, [-2.5,0,0], 3, 8, 0.02, 0.3)
            xml = create_obstacle_grid(xml, [-2.5,0,0], 2, 9, 0.02, 0.3)        
        if options.level==4:
            xml = create_obstacle_grid(xml, [-2.5,0,0], 3, 8, 0.02, 0.3)
            xml = create_obstacle_grid(xml, [-2.5,0,0], 2, 9, 0.02, 0.3)
            xml  = generate_mountain(xml,'0.7 0 0',0.7,1.3,0.02, 0)
        if options.level==5:
            xml = create_obstacle_grid(xml, [-2.5,0,0], 3, 8, 0.02, 0.3)
            xml = create_obstacle_grid(xml, [-2.5,0,0], 2, 9, 0.02, 0.3)
            xml  = generate_mountain(xml,'0.7 0 0',0.7,1.3,0.02, 1)
        if options.level==6:
            xml = create_obstacle_grid(xml, [-2.5,0,0], 3, 8, 0.02, 0.3)
            xml = create_obstacle_grid(xml, [-2.5,0,0], 2, 9, 0.02, 0.3)
            xml  = generate_mountain(xml,'0.7 0 0',0.7,1.3,0.02, 2)

        else:
            pass
        '''
        if options.level==1:            
            xml = create_obstacle_grid(xml, [0,0,0], 3, 3, 0.025, 0.37)
            xml = create_obstacle_grid(xml, [0,0,0], 4, 4, 0.027, 0.37)

        if options.level==2:
            xml = create_obstacle_grid(xml, [0,0,0], 6, 6, 0.027, 0.34)
            xml = create_obstacle_grid(xml, [0,0,0], 7, 7, 0.033, 0.34)

        if options.level==3:
            xml = create_obstacle_grid(xml, [0,0,0], 12, 12, 0.027, 0.3)
            xml = create_obstacle_grid(xml, [0,0,0], 13, 13, 0.033, 0.3)
        
        #Mountains
        if options.level==4:
            xml = create_obstacle_grid(xml, [0,0,0], 12, 12, 0.027, 0.3)
            xml = create_obstacle_grid(xml, [0,0,0], 13, 13, 0.033, 0.3)
            xml  = generate_mountain(xml,'-0.25 0 0',0.6,1.3,0.03, 1) 

        if options.level==5:
            xml = create_obstacle_grid(xml, [0,0,0], 12, 12, 0.027, 0.3)
            xml = create_obstacle_grid(xml, [0,0,0], 13, 13, 0.033, 0.3)
            xml  = generate_mountain(xml,'-0.25 0 0',0.6,1.3,0.031, 2)   

       
        if options.level==6:
            xml = create_obstacle_grid(xml, [0,0,0], 12, 12, 0.027, 0.3)
            xml = create_obstacle_grid(xml, [0,0,0], 13, 13, 0.033, 0.3)
            xml  = generate_mountain(xml,'-0.25 0 0',0.6,1.3,0.032, 3)   

        #Deception
        if options.level==7:
            xml = create_obstacle_grid(xml, [0,0,0], 12, 12, 0.027, 0.3)
            xml = create_obstacle_grid(xml, [0,0,0], 13, 13, 0.033, 0.3)
            xml  = generate_mountain(xml,'-0.25 0 0',0.6,1.3,0.032, 3)   
            xml = generate_deception(xml, level=0)

        
        if options.level==8:
            xml = create_obstacle_grid(xml, [0,0,0], 12, 12, 0.027, 0.3)
            xml = create_obstacle_grid(xml, [0,0,0], 13, 13, 0.033, 0.3)
            xml  = generate_mountain(xml,'-0.25 0 0',0.6,1.3,0.032, 3) 
            xml = generate_deception(xml, level=1)
  

        if options.level==9:
            xml = create_obstacle_grid(xml, [0,0,0], 12, 12, 0.027, 0.3)
            xml = create_obstacle_grid(xml, [0,0,0], 13, 13, 0.033, 0.3)
            xml  = generate_mountain(xml,'-0.25 0 0',0.6,1.3,0.032, 3)
            xml = generate_deception(xml, level=2)'''




       

        for robot in robots:
            xml.worldbody.add('site',
                                name=robot.full_identifier[:-1] + "_start",
                                pos=robot.pos * [1, 1, 0], rgba=[0, 0, 0.1, 1],
                                type="ellipsoid", size=[0.05, 0.05, 0.0001])

        for r in robots:
            for g in r.find_all('geom'):
                if math.isclose(g.size[0], g.size[1]):
                    g.rgba = ".3 0 .3 1"
                else:
                    g.rgba = "0 .3 0 1"

        for hinge in filter(lambda j: j.tag == 'joint', xml.find_all('joint')):
            xml.sensor.add('jointpos', name=f"{hinge.full_identifier}_sensor".replace('/', '_'),
                           joint=hinge.full_identifier)
            
    def process_video_frame(self, frame: np.ndarray):
        v = self.runner.controller.actor_controller.vision
        ratio = .25
        w, h, _ = frame.shape
        raw_vision = v.img
        vision_ratio = raw_vision.shape[0] / raw_vision.shape[1]
        iw, ih = int(ratio * w), int(ratio * h * vision_ratio)
        scaled_vision = cv2.resize(
            cv2.cvtColor(np.flipud(raw_vision), cv2.COLOR_RGBA2BGR),
            (ih,iw),
            interpolation=cv2.INTER_NEAREST
        )
        frame[w-iw:w, h-ih:h] = scaled_vision
