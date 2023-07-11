import math
import numpy as np
from simulation.config import Config
from simulation.my_scenario import Scenario, build_robot
from simulation.my_runner import Runner,RunnerOptions, CallbackType
from simulation.re_runner import ReRunner
from robot.vision import OpenGLVision
from robot.genome import RVGenome
from robot.control import ANNControl
from utils import target_area_distance
from evo_alg.my_map_elite import EvaluationResult
from functools import lru_cache
from typing import Dict, List
from revolve2.core.physics.running import EnvironmentResults
import logging
import math
from functools import lru_cache

logger = logging.getLogger(__name__)




class Evaluator:
    options : RunnerOptions()
    #Hyperparameters
    #Fitness Name
    fitness_function : str
    #Descriptor Names
    feature_names : List = ["distance", "white_gazing"]

    #Scenario Options
    target_position=[2,0,0]
    target_size=0.5
    #Robot Vision
    vision_w: int
    vision_h: int

    #To Store Results
    actor_states = []  
    vision_results = []
    bonus=0

    @classmethod
    def set_runner_options(cls, options: RunnerOptions):
        cls.options = options

    @classmethod
    def set_target_options(cls, w,h):
        cls.vision_w, cls.vision_h=w,h
    
    @classmethod
    def set_descriptors(cls, descriptor_names : List):
        cls.feature_names = descriptor_names

    @classmethod
    def get_result(cls, env_res: EnvironmentResults):
        result = EvaluationResult()
        cls.actor_states = []
        for i in range(len(env_res.environment_states)):
            cls.actor_states.append(env_res.environment_states[i].actor_states)
        result.fitnesses, result.descriptors = cls.fitness(), cls.descriptors()
        return result

    @classmethod
    def _evaluate(cls, genome: RVGenome, options: RunnerOptions, rerun: bool) -> EvaluationResult:
        #Define robot vision dimentions
        robot = build_robot(genome, cls.vision_w, cls.vision_h)
        scenario = Scenario(genome.id())
        scenario.insert_target(cls.target_position, cls.target_size)

        #!!! Create only one runner. either headless or no. No rerun.py e que se escolhe video, view, none
        #Runner, Recebe o scenario, que ja tem o robot e tudo o resto pronto
        if rerun:
            runner = ReRunner(robot, options, scenario.amend, robot_position=Scenario.initial_position(), target_position=cls.target_position, target_size=cls.target_size)
        else:
            runner = Runner(robot, options, Scenario.amend, robot_position=Scenario.initial_position(), target_position=cls.target_position, target_size=cls.target_size)
        
        if cls.options.record is not None:
            runner.callbacks[CallbackType.VIDEO_FRAME_CAPTURED] = scenario.process_video_frame
        
        brain_controller : ANNControl = runner.controller.actor_controller
        brain_controller.vision = \
                OpenGLVision(runner.model, genome.vision, runner.headless)
        
        #Run Simulation
        simulation_result, cls.bonus = runner.run()
        #if cls.bonus>0.0:
            #print("BONUS: ", cls.bonus)

        #Collect viewing behavior
        cls.vision_results = brain_controller.get_robot_vision()

        return cls.get_result(simulation_result)
        
    @classmethod
    def evaluate_evo(cls, genome: RVGenome) -> EvaluationResult:
        return cls._evaluate(genome, cls.options, False)

    @classmethod
    def evaluate_rerun(cls, genome: RVGenome) -> EvaluationResult:
        return cls._evaluate(genome, cls.options, True)
    
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
        descriptors = {}
        if "distance" in Evaluator.feature_names:
            descriptors["distance"] = Evaluator.calculate_distance()
        if "white_gazing" in Evaluator.feature_names:
            descriptors["white_gazing"] = Evaluator.calculate_white_gazing()
        if "avg_speed" in Evaluator.feature_names:
            max_velocity, avg_speed = Evaluator.calculate_velocities()
            descriptors["avg_speed"] = avg_speed
        #dir_changes = Evaluator.calculate_direction_changes()
        #covered_dist = Evaluator.calculate_covered_distance()
        return descriptors
    
    @classmethod
    @lru_cache(maxsize=1)
    def descriptor_bounds(cls):
        bounds=[]
        if "distance" in Evaluator.feature_names:
            distance_bounds = (0, 5.5)
            bounds.append(distance_bounds)
        if "white_gazing" in Evaluator.feature_names:
            white_gazing_bounds = [0, 1]
            bounds.append(white_gazing_bounds)
        if "avg_speed" in Evaluator.feature_names:
            avg_speed_bounds = [0, 4]
            bounds.append(avg_speed_bounds)
        
        max_velocity_bounds = (0, 2)
        dir_changes_bounds = (0, 100)
        covered_dist_bounds = (0, 6)
        return bounds

    @staticmethod
    def descriptor_names():
        return Evaluator.feature_names
    
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
    


        

