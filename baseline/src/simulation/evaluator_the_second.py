import math
import numpy as np
from simulation.config import Config
from simulation.my_scenario import Scenario, build_robot
from simulation.runner import Runner,RunnerOptions, CallbackType
from simulation.re_runner import ReRunner
from robot.vision import OpenGLVision
from robot.genome import RVGenome
from robot.control import ANNControl
from utils import target_area_distance
from evo_alg.my_map_elite import EvaluationResult
from functools import lru_cache
from typing import Dict
from revolve2.core.physics.running import EnvironmentResults
import logging
import math
from functools import lru_cache

logger = logging.getLogger(__name__)




class Evaluator:
    options : RunnerOptions()
    runner : Runner
    #Hyperparameters
    #Fitness Name
    #Descriptor Names

    #-Target
    target_position=[2,0,0]
    target_size=0.5
    #-Robot Vision
    vision_w: int
    vision_h: int

    #To Store Results
    actor_states = []  
    vision_results = []
    bonus=0

    @classmethod
    def set_target_options(cls, w,h):
        cls.vision_w, cls.vision_h=w,h

    @classmethod
    def set_options(cls, options: RunnerOptions):
        cls.options = options


        
    @classmethod
    def get_result(cls, env_res: EnvironmentResults):
        cls.actor_states = []
        for i in range(len(env_res.environment_states)):
            cls.actor_states.append(env_res.environment_states[i].actor_states)
        return cls.fitness(), cls.descriptors()

    @classmethod
    def _evaluate(cls, genome: RVGenome, options: RunnerOptions) -> EvaluationResult:
        #Define robot vision dimentions
        robot = build_robot(genome, cls.vision_w, cls.vision_h)
        #for _ in range(2):
        scenario = Scenario(genome.id())
        scenario.insert_target(cls.target_position, cls.target_size)

        runner = Runner(robot, options, Scenario.amend)
        
        brain_controller : ANNControl = runner.controller.actor_controller
        brain_controller.vision = \
                OpenGLVision(runner.model, genome.vision, runner._headless)

        simulation_result, cls.bonus = runner.run()
        #if cls.bonus>0.0:
            #print("BONUS: ", cls.bonus)
        cls.vision_results = brain_controller.get_robot_vision()

        result = EvaluationResult()
        result.fitnesses, result.descriptors = cls.get_result(simulation_result)

        return result
        
    @classmethod
    def evaluate_evo(cls, genome: RVGenome) -> EvaluationResult:
        return cls._evaluate(genome, cls.options)

    
    @staticmethod
    def fitness2() -> Dict[str, float]:
        #Final distance to the square area 
        di = target_area_distance(Evaluator.actor_states[0].position, Evaluator.target_position, Evaluator.target_size)
        df = target_area_distance(Evaluator.actor_states[-1].position, Evaluator.target_position, Evaluator.target_size)
        closeness_score = 100 - (df * 100) / di
        score = closeness_score + Evaluator.bonus
        return {"closeness": score}
    
    def fitness()-> Dict[str, float]:
        score=0
        x=[]
        for view in Evaluator.vision_results:
            x.append(sum(view))
        brightest=max(x)
        score = brightest/(Evaluator.vision_w*Evaluator.vision_h)
        return{"brightness": 100*score}
    
    @classmethod
    @lru_cache(maxsize=1)
    def fitness_bounds(cls):
        min_max = [0, 100]
        return [tuple(min_max)]

    @staticmethod
    def fitness_name():
        return "brightness"
    

    #############################################################################
    #            DESCRIPTORS
    #############################################################################
    
    @staticmethod
    def descriptors() -> Dict[str, float]:
        distance = Evaluator.calculate_distance()
        max_velocity, avg_speed = Evaluator.calculate_velocities()
        dir_changes = Evaluator.calculate_direction_changes()
        covered_dist = Evaluator.calculate_covered_distance()
        white_gazing = Evaluator.calculate_white_gazing()
        #print("white_gazing", white_gazing)
        #print("distance", round(distance,2))
        #print("covered_dist", round(covered_dist, 2))
        #print("max_velocity", max_velocity)
        #print("dir_changes",dir_changes)
        return {"distance": distance, "white_gazing": white_gazing }
    
    @classmethod
    @lru_cache(maxsize=1)
    def descriptor_bounds(cls):
        max_velocity_bounds = (0, 2)
        distance_bounds = (0, 5.5)
        dir_changes_bounds = (0, 100)
        covered_dist_bounds = (0, 6)
        white_gazing_bounds = [0, 1]
        return [distance_bounds, white_gazing_bounds]

    @staticmethod
    def descriptor_names():
        return ["distance", "white_gazing"]
    
    @staticmethod
    def calculate_distance():
        return float(((Evaluator.actor_states[-1].position[0] - Evaluator.actor_states[0].position[0]) ** 2 +
                      (Evaluator.actor_states[-1].position[1] - Evaluator.actor_states[0].position[1]) ** 2) ** 0.5)

    @staticmethod
    def calculate_covered_distance():
        positions = [state.position for state in Evaluator.actor_states]
        total_distance = 0.0
        for i in range(len(positions) - 1):
            current_pos = positions[i][:2]  # Ignore the z-coordinate
            next_pos = positions[i + 1][:2]
            
            distance = math.sqrt((next_pos[0] - current_pos[0]) ** 2 + (next_pos[1] - current_pos[1]) ** 2)
            total_distance += distance
        
        return total_distance
        
    @staticmethod
    def calculate_velocities():
        positions = [state.position for state in Evaluator.actor_states]
        time_step = Config.sampling_frequency 
        # (0.011)
        #time_step = 0.001
        velocities = []
        for i in range(len(positions) - 1):
            x1, y1, _ = positions[i]
            x2, y2, _ = positions[i + 1]
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            velocity = distance / time_step
            velocities.append(velocity)
        average_velocity = sum(velocities) / len(velocities)
        max_velocity = max(velocities)
        return max_velocity, average_velocity
        
    @staticmethod
    def calculate_white_gazing():
        w,h = Evaluator.vision_w, Evaluator.vision_h
        for view in Evaluator.vision_results:
            if   (w*h)*.75<=sum(view)<=(w*h):
                score=1
            elif (w*h)*.5<=sum(view)<=(w*h)*.75:
                score=0.75
            elif (w*h)*.25<=sum(view)<=(w*h)*.5:
                score=0.5
            elif (w*h)*0.125<=sum(view)<=(w*h)*.25:
                score=0.25
            else:
                score=0
        return score


        '''Summing all pixels values, and divide by total number of pixels
        flat_vision = [item for sublist in Evaluator.vision_results for item in sublist]
        #print(flat_vision)
        print("score",sum(flat_vision)/len(flat_vision))

        PROBLEM, It varies between 0 amd 1, but values above 0.5 would be rare,
        since the robot can't keep being looking, he has to move.  
        '''

    @staticmethod
    def trajectory_description():
        positions = np.array([state.position for state in Evaluator.actor_states])
        # Extract the y coordinates into a separate array
        y_coordinates = positions[:, 1]
        # Find the maximum and minimum y values
        max_y = np.max(y_coordinates)
        min_y = np.min(y_coordinates)
        print("min_y:",abs(min_y)," | max_y:",max_y)
        if max_y>abs(min_y):
            return max_y
        else:
            return min_y


    @staticmethod
    def calculate_direction_changes():
        positions = [state.position for state in Evaluator.actor_states]
        prev_direction = None
        direction_changes = 0
        
        for i in range(len(positions) - 1):
            current_pos = positions[i]
            next_pos = positions[i + 1]
            
            # Calculate the angle between the current and next positions
            angle = math.atan2(next_pos[1] - current_pos[1], next_pos[0] - current_pos[0])
            
            # Determine the direction based on the angle
            '''if angle >= -math.pi / 6 and angle <= math.pi / 6:
                direction = 'going straight'
            elif angle > math.pi / 6 and angle < math.pi / 3:
                direction = 'going straight right'
            elif angle >= math.pi / 3 and angle <= 2 * math.pi / 3:
                direction = 'going back right'
            elif angle > 2 * math.pi / 3 or angle < -2 * math.pi / 3:
                direction = 'going back'
            elif angle >= -2 * math.pi / 3 and angle <= -math.pi / 3:
                direction = 'going back left'
            else:
                direction = 'going straight left' '''
            
            if angle >= -math.pi / 4 and angle <= math.pi / 4:
                direction = 'straight'
                if angle >= 0:
                    direction += ' right'
                else:
                    direction += ' left'
            elif angle > math.pi / 4 and angle < 3 * math.pi / 4:
                direction = 'back right'
            elif angle >= 3 * math.pi / 4 or angle <= -3 * math.pi / 4:
                direction = 'back left'
            else:
                direction = 'straight left'
            
            # Check if there is a change in direction
            if prev_direction is not None and direction != prev_direction:
                direction_changes += 1
            
            prev_direction = direction
            
        direction_change_percentage = (direction_changes / (len(positions) - 1)) * 100
        return direction_change_percentage
    


        

