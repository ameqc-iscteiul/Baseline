import math
import numpy as np
from simulation.config import Config
from simulation.my_scenario import Scenario, build_robot
from simulation.runner import Runner, RunnerOptions, CallbackType
from robot.vision import OpenGLVision
from robot.genome import RVGenome
from robot.control import ANNControl
from utils import target_area_distance
from evo_alg.my_map_elite import EvaluationResult
from functools import lru_cache
from typing import Dict, List, Tuple
from revolve2.core.physics.running import EnvironmentResults
import logging
import math
import abrain
logger = logging.getLogger(__name__)


'''Object Responsible for the Evaluations. Only concerned with one evaluation'''
class Evaluator:
    runner_options = RunnerOptions()

    #Fitness Name
    fitness_function : str
    features = []
    #Robot 
    vision_w: int
    vision_h: int
    robot_type:int

    #To Store Results
    actor_states = []  
    vision_results = []
    bonus=0
    ann = None

    @classmethod
    def set_options(cls, descriptor_names, fitness_name, robot_type, vision_w,vision_h, level):
        cls.vision_w, cls.vision_h = vision_w, vision_h
        cls.robot_type = robot_type
        cls.features = descriptor_names
        cls.fitness_function = fitness_name
        cls.runner_options.level=level


    @classmethod
    def set_runner_options(cls, options: RunnerOptions):
        cls.runner_options = options
    
    @classmethod
    def set_level(cls, level: int):
        cls.runner_options.level = level
        logger.warning(f"Change to level {cls.runner_options.level}")

    @classmethod
    def set_view_dims(cls, w,h):
        cls.vision_w, cls.vision_h=w,h
    
    @classmethod
    def set_descriptors(cls, descriptor_names : List):
        cls.features = descriptor_names
    
    @staticmethod
    def _clip(values: Dict[str, float], bounds: List[Tuple[float]], name: str):
        for i, b in zip(values.items(), bounds):
            k, v, lower, upper = *i, *b
            if not lower <= v <= upper:
                values[k] = max(lower, min(v, upper))
                logger.warning(f"Out-of-bounds value {name},{k}: {v} not in [{lower},{upper}]")
        return values

    @classmethod
    def get_result(cls, env_res: EnvironmentResults):
        result = EvaluationResult()
        cls.actor_states = []
        for i in range(len(env_res.environment_states)):
            cls.actor_states.append(env_res.environment_states[i].actor_states)
        result.fitnesses = cls.fitness() 
        result.descriptors = cls._clip(cls.descriptors(), cls.descriptor_bounds(),"features")
        return result
    
    @classmethod
    def get_ann(cls):
        return cls.ann

    
    @classmethod
    def _evaluate(cls, genome: RVGenome, options: RunnerOptions) -> EvaluationResult:
       
        robot = build_robot(genome, cls.vision_w, cls.vision_h , cls.robot_type)
        scenario = Scenario()
        scenario.insert_target(options.target_position, options.target_size)
        
        runner = Runner(robot, options, scenario.amend)
        if cls.runner_options.record is not None:
            scenario.assign_runner(runner)
            runner.callbacks[CallbackType.VIDEO_FRAME_CAPTURED] = scenario.process_video_frame
        
        brain_controller : ANNControl = runner.controller.actor_controller
        #To think: 
        # Maybe the Vision could have more definition, but then its converted to a smaller w*h img
        # which then is gonna be used to feed the lum_input of the ANN
        brain_controller.vision = \
                OpenGLVision(runner.model, (cls.vision_w, cls.vision_h), runner.headless)
        #Run Simulation
        simulation_result, cls.bonus = runner.run()

        #Save ANN to plot
        cls.ann=brain_controller.get_ANN()

        #Collect viewing behavior
        cls.vision_results = brain_controller.get_robot_vision()
        return cls.get_result(simulation_result)
        
    @classmethod
    def evaluate(cls, genome: RVGenome) -> EvaluationResult:
        return cls._evaluate(genome, cls.runner_options)

    @classmethod
    def evaluate_rerun(cls, genome: RVGenome) -> EvaluationResult:
        return cls._evaluate(genome, cls.runner_options)
    
    def fitness()-> Dict[str, float]:
        if Evaluator.fitness_function == "brightness":
            #Corresponds to the brightest point seen
            x=[]
            for view in Evaluator.vision_results:
                x.append(sum(view))
            brightest=max(x)
            score = (brightest*100/(Evaluator.vision_w*Evaluator.vision_h)+ Evaluator.bonus)
            if score > 110 : score = 110
            return{"brightness": score }
        
        elif Evaluator.fitness_function == "new_fitness":
            x=[]
            white_gazes=0
            for view in Evaluator.vision_results:
                #calculate avg grayscale value per view
                avg = sum(view)/len(view)
                x.append(avg)
                if avg >= 0.96:
                #Avg above this value count as full white gazes
                    white_gazes+=1
            brightest=max(x)
            score = brightest*80 + white_gazes + Evaluator.bonus
            if score > 110 : score = 110
            return{"new_fitness": score }
        

        elif Evaluator.fitness_function == "count_white_pixels":
            view_scores=[]
            i=1
            for view in Evaluator.vision_results:
                #Select the white pixels from view
                white_pixels = [value for value in view if value == 1]
                #If there are white pixels:
                if len(white_pixels) != 0:
                    #Calculate the Percentage of white pixels per view
                    view_score=(len(white_pixels)/len(view))*i
                    view_scores.append(view_score)
                i+=(1/len(Evaluator.vision_results))
            #Calculate avg Percentage of white pixels per view
            avg_view_score = (sum(view_scores)/len(Evaluator.vision_results))
            score = avg_view_score*100 + Evaluator.bonus
            
            #score = white_pixels + Evaluator.bonus
            if score > 110 : score = 110
            return{"count_white_pixels": score }
        
        elif Evaluator.fitness_function == "stareness":
            #Corresponds to the avg brightest from the last 3 views
            n=3
            #check the last n gazes amount of white seen
            white_total=sum([value for view in Evaluator.vision_results[-n:] for value in view])
            max_white_poss = len(Evaluator.vision_results[0])*n
            
            score = (white_total*100/max_white_poss) + Evaluator.bonus
            if score > 110 :
                print("Fitness higher the 110")
                score = 110
            return{"stareness": score }
        
        elif Evaluator.fitness_function == "new":
            x=[]
            for view in Evaluator.vision_results:
                #calculate avg grayscale value per view
                avg = sum(view)/len(view)
                x.append(avg)
                #If an avg is 1, the view was fully white

            '''full_white_views=0
            if max(x)==1:
                full_white_views = x.count(1)'''
            #1-score = max(x)*75 + full_white_views + Evaluator.bonus

            '''last_view = x[-1]'''
            #2-score = max(x)*60 + last_view*30 + full_white_views + Evaluator.bonus

            #Fitness (60 brightest + 30 last view)=90% if they see a full gaze, and one at the end. 
            #Extra points for the amount of full whites they seen. Bonus for the time they saved 
            
            #3
            brightest_view=max(x)
            n=4
            #Similar to the stop condition in Runner
            last_n_avgs = x[-n:]
            last_n_views_avg_brightness = sum(last_n_avgs)/len(last_n_avgs)
           
            score = brightest_view*90 + last_n_views_avg_brightness*10 + Evaluator.bonus

            #If it gets at least one full white gaze : brightest_view=1 we get 60%
            #If the last n views are fully white : we will get the other 40%, reaching 100% score!
            #If they do that in less time, they get a time bonus!

                    
            return{"new": score }
        else:
            logger.warning(f"Invalid Fitness Function Name {Evaluator.fitness_function}")
        

    
    @classmethod
    @lru_cache(maxsize=1)
    def fitness_bounds(cls):
        min_max = [0, 110]
        return [tuple(min_max)]

    @staticmethod
    def fitness_name():
        return Evaluator.fitness_function
    

    #############################################################################
    #            DESCRIPTORS
    #############################################################################
    
    @staticmethod
    def descriptors() -> Dict[str, float]:
        descriptors = {}
        for i in range(2) :
            if Evaluator.features[i] == "distance":
                descriptors["distance"] = Evaluator.calculate_distance()
            elif Evaluator.features[i] =="white_gazing":
                descriptors["white_gazing"] = Evaluator.calculate_Avg_View()
            elif Evaluator.features[i] =="avg_speed":
                max_velocity, avg_speed = Evaluator.calculate_velocities()
                descriptors["avg_speed"] = avg_speed
            elif Evaluator.features[i] =="trajectory":
                descriptors["trajectory"] = Evaluator.trajectory_description()

        #dir_changes = Evaluator.calculate_direction_changes()
        #covered_dist = Evaluator.calculate_covered_distance()
        return descriptors
    
    @classmethod
    def descriptor_bounds(cls):
        bounds=[]
        for i in range(2) :
            if cls.features[i] == "distance":
                distance_bounds = (0, 5.5)
                bounds.append(distance_bounds)
            elif cls.features[i] =="white_gazing":
                white_gazing_bounds = [0.05, 0.4]
                bounds.append(white_gazing_bounds)
            elif cls.features[i] =="avg_speed":
                avg_speed_bounds = [0.05, 0.6]
                bounds.append(avg_speed_bounds)
            elif cls.features[i] == "trajectory":
                trajectory_bounds = [-2.5,2.5]
                bounds.append(trajectory_bounds)

        max_velocity_bounds = (0, 2)
        dir_changes_bounds = (0, 100)
        covered_dist_bounds = (0, 6)

        return bounds
    
    @staticmethod
    def descriptor_names():
        return Evaluator.features
    
    @staticmethod
    def calculate_distance():
        return float(((Evaluator.actor_states[-1].position[0] - Evaluator.actor_states[0].position[0]) ** 2 +
                      (Evaluator.actor_states[-1].position[1] - Evaluator.actor_states[0].position[1]) ** 2) ** 0.5)

    
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
    def calculate_Avg_View():
        avgs = []
        for view in Evaluator.vision_results:
            #Collect the avg of every View
            avgs.append(sum(view)/len(view))
        #Calculate avg of avgs
        final_score = (sum(avgs)/len(avgs))
        return final_score 
    

    @staticmethod
    def calculate_white_gazing():
        view_area = Evaluator.vision_w * Evaluator.vision_h
        total_score = 0
        total_sum = 0
        for view in Evaluator.vision_results:
            score = sum(view)/view_area
            total_score += score * sum(view)
            total_sum += sum(view)
        if total_sum > 0:
            final_score = total_score / total_sum
        else:
            final_score = 0
        print ('white gaze score',round(final_score,3))
        return final_score

    @staticmethod
    def trajectory_description():
        positions = np.array([state.position for state in Evaluator.actor_states])
        # Extract the y coordinates into a separate array
        y_coordinates = positions[:, 1]
        # Find the maximum and minimum y values
        max_y = np.max(y_coordinates)
        min_y = np.min(y_coordinates)
        #print("min_y:",abs(min_y)," | max_y:",max_y)
        if max_y>abs(min_y):
            return max_y
        else:
            return min_y
        
    @staticmethod
    def trajectory_descriptionZ():
        positions = np.array([state.position for state in Evaluator.actor_states])
        # Extract the y coordinates into a separate array
        z_coordinates = positions[:, 2]
        # Find the maximum and minimum y values
        max_z = np.max(z_coordinates)
        min_z = np.min(z_coordinates)
        #print("min_y:",abs(min_z)," | max_y:",max_z)
        if max_z>abs(min_z):
            return max_z
        else:
            return min_z


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
    
    @staticmethod
    def calculate_covered_distance():
        #Sums the distance between all sampled states. Calculating the lenght of the full tragectory
        positions = [state.position for state in Evaluator.actor_states]
        total_distance = 0.0
        for i in range(len(positions) - 1):
            current_pos = positions[i][:2] # Ignore the z-coordinate
            next_pos = positions[i + 1][:2]
            distance = math.sqrt((next_pos[0] - current_pos[0]) ** 2 + (next_pos[1] - current_pos[1]) ** 2)
            total_distance += distance
        
        return total_distance
    
    @staticmethod
    def fitness2() -> Dict[str, float]:
        #Final distance to the square area 
        di = target_area_distance(Evaluator.actor_states[0].position, Evaluator.target_position, Evaluator.target_size)
        df = target_area_distance(Evaluator.actor_states[-1].position, Evaluator.target_position, Evaluator.target_size)
        closeness_score = 100 - (df * 100) / di
        score = closeness_score + Evaluator.bonus
        return {"closeness": score}
    


        

