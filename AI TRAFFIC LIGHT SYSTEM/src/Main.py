#BUKASA MUYOMBO
#AI TRAFFIC CONTROL SYSTEM
#PEDESTRIANS = PURPLE
#VEHICLES = Others, there are trucks cars and busses, from the dataset


#When the traffic light is green, vehicles have the right of way, and when it is red, pedestrians have the right of way

import tkinter as tk
import random
import pandas as pd
from datetime import datetime, timedelta
import math
import numpy as np
import datetime

#read the dataset
#from kaggle
df = pd.read_csv('C:/Users/User/Desktop/HYP/D07/data/Traffic.csv') 

# Convert Time to datetime to control when they are added
#from a string in this format 12:34:56 PM
df['Time'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p')

# Sort by Time in ascending order
df = df.sort_values('Time')
#SensorData collects data from the environment for sensor inputs.

def getTrafficData(time):
    # find the closest time in the dataset
    closestTime = min(df['Time'], key=lambda x: abs(x - time))
    
    # Get the row for the closest time
    row = df[df['Time'] == closestTime].iloc[0] 
    #iloc[0] gets the first row
    
    # # # # # #
    # most are types of vehicles
    return {
        'time': row['Time'],
        'cars': row['CarCount'],
        'bikes': row['BikeCount'],
        'buses': row['BusCount'],
        'trucks': row['TruckCount'],
        'pedestrians': row['PedestrianCount']
    }

vehicle_type_weights = {
            'car': 1.0,    # base weight
            'bike': 0.5,   # less importance
            'bus': 1.5,    # higher importance
            'truck': 1.5,  # higher importance
            'default': 1.0 # default for undefined types
        }


class SensorData:
    def __init__(self, environment):
        self.environment = environment

    def collectData(self):
        sensorData = []
        for y in range(self.environment.height):
            for x in range(self.environment.width):
                obj = self.environment.grid[y][x]
                if isinstance(obj, Vehicle):
                    sensorData.append(('vehicle', x, y))
                elif isinstance(obj, Pedestrian):
                    sensorData.append(('pd', x, y))
        #print(sensorData)
        return sensorData

#objectDetector detects vehicles and pedestrians based on sensor data
class ObjectDetector:
    def __init__(self, sensorData):
        self.sensorData = sensorData
        

    def detectVehicles(self):
        # for every element (data) in sensordata
        #if the first element of data is a string 'vehicles'
        #then add it starting from the second element, and remove the string
        #so that it is not redundant
        vehicles = [data[1:] for data in self.sensorData if data[0] == 'vehicle']
        #print(vehicles)
        return vehicles

    def detectpds(self):
        pds = [data[1:] for data in self.sensorData if data[0] == 'pd']
        #print(pds)
        return pds

#
#TrafficLightSystem manages the traffic lights at different intersections. 
class TrafficLightSystem:
    def __init__(self, root, location, environment):
        self.root = root
        self.location = location
        self.trafficLights = []
        self.environment = environment
        self.aiCrl = AIController(self, environment)
        self.ucbCrl = AIControllerUCB(self, environment, alpha=0.7, gamma=0.1, c=0.1)
        self.bmrl = AIControllerBoltmann(self, environment, alpha=0.2, gamma=0.5, tau=0.10)
        self.dCrl = DeterministicController(self, environment, interval=20)
        
    def addTrafficLight(self, trfLight):
        self.trafficLights.append(trfLight)

    def removeTrafficLight(self, trfLight):
        self.trafficLights.remove(trfLight)

    def changeTrafficLight(self):
        for light in self.trafficLights:
            oldColor = light.color
            decision = self.aiCrl.decideAction(light)
            #decision = self.ucbCrl.decideAction(light)
            #decision = self.bmrl.decideAction(light)
            #decision = self.dCrl.decideAction(light)
            light.changeColor(decision)
            
            if oldColor != light.color:
                if light.color == 'green':
                    print(f"Traffic Light {light.id} at ({light.x},{light.y}) changed to GREEN. Vehicles may proceed.")
                elif light.color == 'red':
                    print(f"Traffic Light {light.id} at ({light.x},{light.y}) changed to RED. Pedestrians may proceed.")
                elif light.color == 'yellow':
                    print(f"Traffic Light {light.id} at ({light.x},{light.y}) changed to YELLOW. Proceed with caution.")
        
        self.root.after(200, self.changeTrafficLight)

    
##TrafficLight represents a traffic light with methods to change its color and get its status.
class TrafficLight:
    def __init__(self, id, color, x, y, status):
        self.id = id
        self.color = color
        self.x = x
        self.y = y
        self.status = status
        self.transition_time = 0
        self.transition_duration = 5  
        #the transition time
        # nnumber of frames to show yellow
        self.nxtColor = None

    def changeColor(self, new_color):
        #if next color was known, this block would be skipped
        if new_color != self.color and self.nxtColor is None:
            self.nxtColor = new_color
            self.transition_time = self.transition_duration
        elif self.nxtColor:
            if self.transition_time > 0:
                self.color = 'yellow'
                self.transition_time -= 1
            else:
                self.color = self.nxtColor
                self.nxtColor = None
        
    def getStatus(self):
        return self.status

        

#AIController controls the traffic lights based on sensor data and rewards
class AIController:
    def __init__(self, trfLightSystem, environment, alpha=0.2, gamma=0.9, epsilon=0.4):
        self.trfLightSystem = trfLightSystem
        self.environment = environment
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # the discount factor
        self.epsilon = epsilon  # the exploration rate
        self.Qtable = {}  # q table for storing action values
        self.decCoolDown = 10  # Minimum frames between decisions
        self.currentCooldown = 0
        self.vehicle_arrival_times = {}
        #epsilon = 0.9*epsilon
        

    #determines the state of the traffic light based on sensor data
    def getState(self, trfLight, sensorData):
        #  use a collection of (numVehicles, numbPedestrinas)
        objDetector = ObjectDetector(sensorData)
        decVehicls = objDetector.detectVehicles()
        decPeds = objDetector.detectpds()

        numVehicles = sum(1 for vehicle in decVehicls if vehicle[0] == trfLight.x - 1 and vehicle[1] == trfLight.y)
        numbPedestrinas = sum(1 for pd in decPeds if pd[1] == trfLight.y - 1 and pd[0] == trfLight.x)
        state = (numVehicles, numbPedestrinas)
        return state
    
    #also determines what to do based on the current score
    def getAction(self, state):
        #uses the epsilon greedy algorithm
        # decisions to xplore or exploit based on the epsilon value
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(['green', 'red'])  # explore choice
        else:
            if state in self.Qtable:
                action = max(self.Qtable[state], key=self.Qtable[state].get)  # exploit choice
            else:
                action = 'red'  # set the default value to green to allowe for the pedestrians to have priority

        return action

    def updateQTable(self, state, action, reward, nxtState):
        #updates teh q table based on the states
        if state in self.Qtable:
            if action in self.Qtable[state]:
                qVal = self.Qtable[state][action]
            else:
                qVal = 0
        else:
            qVal = 0
        # Q(S,A)t+1 = Q(S,A) + alpha((r + g)*(max_previous_q_value - current )
        nxtMaxQV = max(self.Qtable[nxtState].values()) if nxtState in self.Qtable else 0
        newQVal = qVal + self.alpha * (reward + self.gamma * nxtMaxQV - qVal)

        if state not in self.Qtable:
            self.Qtable[state] = {}
            #inserts this new qvalue back into that slot
        self.Qtable[state][action] = newQVal

        """
        Qtable example = {
                ("Green", "(10 cars, 3 pedestrians)"): {
                    "Change to Green": 0.5,
                    "Change to Yellow": 0.2,
                    "Change to Red": -0.1
                 })
                 """

    #choses what to do next after getting the actions and the states
    def decideAction(self, trfLight):
        if self.currentCooldown > 0:
            self.currentCooldown -= 1
            return trfLight.color
        #gate the state
        sensorData = SensorData(self.environment).collectData()
        state = self.getState(trfLight, sensorData)
        action = self.getAction(state)

        self.environment.updateEnvironment()  # Update the environment after action
        nextState = self.getState(trfLight, sensorData)
        reward = self.getReward(trfLight, simulation_time = None)

        # Update the Q-table
        self.updateQTable(state, action, reward, nextState)
        
        self.currentCooldown = self.decCoolDown
        return action

    def getReward(self, trfLight, simulation_time = None):
        if simulation_time is None:
            simulation_time = datetime.datetime.now()

        # look at the relativ positions and numbers of pds and vehicles that have passed the traffic light
        numVehiclePassed = sum(1 for vehicle in self.environment.vehicles if vehicle.x == trfLight.x and vehicle.y == trfLight.y)
        #count vehicles waiting directly before the traffic light
        numVehiclesWaiting = sum(1 for vehicle in self.environment.vehicles if vehicle.x == trfLight.x - 1 and vehicle.y == trfLight.y)
        #count thr vrhivles that are back from the traffic light() more than 1 unit behind
        numVehiclesFAR = sum(1 for vehicle in self.environment.vehicles if vehicle.x < trfLight.x - 1 and vehicle.y == trfLight.y)
        # we couount pedestrians waitiing 
        numbpdsWaiting = sum(1 for pd in self.environment.pds if pd.y == trfLight.y - 1 and pd.x == trfLight.x)
        #count the number of pedestrians waiting far from the crosswalk (further than 1 unit)
        numbPedestrinasfar = sum(1 for pd in self.environment.pds if pd.y < trfLight.y - 1 and pd.x == trfLight.x)

        # give weights to different factors( vehicles, pedestrians waiting, based on importance)
        vWeight = 10 #base weight for vehicles
        pWeight = 0.1 # base weight for pedestrians
        waitingWait = 5#weight for vehilces waiting close to the light
        farWeight = 0.05 # and far from the light

        # Adjust penalty based on vehicle types
        vehicleScore = 0
        for vehicle in self.environment.vehicles:
            #if it is waiting
            if vehicle.x == trfLight.x - 1 and vehicle.y == trfLight.y:
                vehicle_type = getattr(vehicle, 'type', 'default')  # Get vehicle type or default
                type_weight = vehicle_type_weights.get(vehicle_type, vehicle_type_weights['default'])
                vehicleScore += waitingWait * type_weight  # Apply weight to the waiting penalty


        #dynamic weights adjustment for rush hour
    
        for vehicle in self.environment.vehicles:
            if vehicle.x == trfLight.x - 1 and vehicle.y == trfLight.y:
                if vehicle not in self.vehicle_arrival_times:
                    self.vehicle_arrival_times[vehicle] = simulation_time

        is_rush_hour = 7 <= simulation_time.hour <= 9 or 17 <= simulation_time.hour <= 19
        traffic_density = numVehiclesWaiting + numVehiclesFAR + numbpdsWaiting + numbPedestrinasfar
        vWeight, pWeight = self.get_dynamic_weights( is_rush_hour, traffic_density)

        waitingWait = 5
        farWeight = 0.05

        # the weighted sum of different factors
        # penalize vehicles waiting and those far away by applying the weights
        vehicleScore = (numVehiclesWaiting * waitingWait + numVehiclesFAR * farWeight) * vWeight
        pedestianScore = (numbpdsWaiting * waitingWait + numbPedestrinasfar * farWeight) * pWeight

        # raward for letting vehicles pass and penalize for waiting pds
        # higher score if more vehicles pass and fewer penalties for waiting vehicles/pedestrians
        reward = numVehiclePassed - vehicleScore - pedestianScore

        current_waiting_vehicles = [v for v in self.environment.vehicles if v.x == trfLight.x - 1 and v.y == trfLight.y]
        vehicle_wait_times = self.calculate_vehicle_wait_times(current_waiting_vehicles, simulation_time)


        # New penalties
        # 1. wait time penalty
        max_vehicles_wait = 30  # Maximum wait time for vehicles in seconds
        max_pedestrian_wait = 10  # Maximum wait time for pedestrians in seconds
        wait_time_penalty = self.calculate_wait_time_penalty(
            vehicle_wait_times,
            [p for p in self.environment.pds if p.y == trfLight.y - 1 and p.x == trfLight.x],
            max_vehicles_wait,
            max_pedestrian_wait
        )
        reward -= wait_time_penalty

        # safety penalty for pedestrians
        max_safe_wait_time = 15  # Maximum safe wait time for pedestrians in seconds
        pedestrians_waiting = [p for p in self.environment.pds if p.y == trfLight.y - 1 and p.x == trfLight.x]
        safety_penalty = self.calculate_safety_penalty(pedestrians_waiting, max_safe_wait_time)
        reward -= safety_penalty

        # congestion penalty
        congestion_threshold = 10  # Number of vehicles/pedestrians that indicate congestion
        congestion_penalty = max(0, (numVehiclesWaiting + numbpdsWaiting - congestion_threshold) * 0.5)
        reward -= congestion_penalty

        # frequent switching penalty, so that it doesnt just switch lights every short period
        #it must actually observe
        if hasattr(self, 'last_action') and self.last_action != trfLight.color:
            switching_penalty = 2
            reward -= switching_penalty
        self.last_action = trfLight.color

        self.clean_up_vehicle_arrival_times(trfLight)

        return reward

    def clean_up_vehicle_arrival_times(self, trfLight):
        # we remove vehicles that have passed the traffic light from vehicle_arrival_times
        self.vehicle_arrival_times = {v: t for v, t in self.vehicle_arrival_times.items() if not (v.x == trfLight.x and v.y == trfLight.y)}


    # pedestrian and vehivle wait time adjustments
    def calculate_wait_time_penalty(self, vehicle_wait_times, pedestrians_waiting, max_vehicles_wait, max_pedestrian_wait):
        vehicle_wait_penalty = sum(min(wait_time, max_vehicles_wait) for wait_time in vehicle_wait_times)
        pedestrian_wait_penalty = sum(min(getattr(pedestrian, 'wait_time', 0), max_pedestrian_wait) for pedestrian in pedestrians_waiting)
        #give more importance to them
        pedestrian_penalty_weight = 0.6
        vehicle_penalty_weight = 0.5
        
        return (vehicle_penalty_weight * vehicle_wait_penalty) + (pedestrian_penalty_weight * pedestrian_wait_penalty)

    def calculate_vehicle_wait_times(self, current_waiting_vehicles, current_time):
        wait_times = []
        for vehicle in current_waiting_vehicles:
            if vehicle in self.vehicle_arrival_times:
                wait_time = (current_time - self.vehicle_arrival_times[vehicle]).total_seconds()
                wait_times.append(wait_time)
        return wait_times

    #DYNAMIC WEIGHTS ASSIGNMENT BASED ON TRAFFIC CONDITIONS
    def get_dynamic_weights(self, is_rush_hour, traffic_density):
        #check if it is rush hour or if traffic density is above a certain threshold
        if is_rush_hour or traffic_density > 20:
           #prioritize vehicles during rush hour by giveing them a higher weight to make traffic flow more smoothly
            vehicle_weight = 1.5 #increase importance of
            pedestrian_weight = 1.0

        else:
            #if it is not rush hour, prioritize the pedestrians
            vehicle_weight = 1.0
            pedestrian_weight = 1.5

        return vehicle_weight, pedestrian_weight
    

    # SAFETY EMPHASIS FOR PEDESTRIAN WAITING TIMES
    def calculate_safety_penalty(self,pedestrians_waiting, max_safe_wait_time):
        # we initialize the safety_penalty at time 0
        safety_penalty = 0

        for pedestrian in pedestrians_waiting:
            #we iterate over each pedestrian that is waiting
            #if a pedestrians waiting time exceeds the safe limit, we apply a heavy pen
            if pedestrian.wait_time > max_safe_wait_time:
                #add a penalty for every second over the safe wait time
                safety_penalty += (pedestrian.wait_time - max_safe_wait_time) * 2

        return safety_penalty


class AIControllerUCB:
    def __init__(self, trfLightSystem, environment, alpha=0.2, gamma=0.5, c=2.0):
        self.trfLightSystem = trfLightSystem
        self.environment = environment
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.c = c  # exploration constant
        self.Qtable = {}  # Q table for action values
        self.action_count = {}  # stores how many times each action is taken
        self.decCoolDown = 10  # minimum frames between decisions
        self.currentCooldown = 0
        self.total_action_count = {}
        self.vehicle_arrival_times = {}

    # get the current state based on traffic and pedestrian data
    #same as aicontroller
    def getState(self, trfLight, sensorData):
        objDetector = ObjectDetector(sensorData)
        decVehicls = objDetector.detectVehicles()
        decPeds = objDetector.detectpds()

        numVehicles = sum(1 for vehicle in decVehicls if vehicle[0] == trfLight.x - 1 and vehicle[1] == trfLight.y)
        numbPedestrinas = sum(1 for pd in decPeds if pd[1] == trfLight.y - 1 and pd[0] == trfLight.x)
        state = (numVehicles, numbPedestrinas)
        return state
    
    # UCB action selection based on the current state
    def getAction(self, state):
        # Ensure the state exists in the Q table and action count
        #if q table is not emptty
        if state not in self.Qtable:
            self.Qtable[state] = {'green': 0, 'red': 0}  # initialize Q-values for both actions
            self.action_count[state] = {'green': 0, 'red': 0}  # initialize action counts

        # Calculate the total number of times any action has been taken in this state
        total_count = sum(self.action_count[state].values())
        #store it here
        self.total_action_count[state] = total_count if total_count > 0 else 1 
         # Avoid division by zero

        # UCB formula: q(s,a) + c * sqrt(ln(total count) / N(s,a))
        ucb_values = {}
        for action in self.Qtable[state]:
            action_count = self.action_count[state][action]
            if action_count == 0:
                ucb_values[action] = float('inf')  # assign a large value to encourage exploration of unselected actions
            else:
                ucb_values[action] = self.Qtable[state][action] + self.c * math.sqrt(
                    math.log(self.total_action_count[state]) / action_count
                )

        # Choose the action with the highest UCB value
        best_action = max(ucb_values, key=ucb_values.get)
        return best_action

    # update Q table after taking an action
    def updateQTable(self, state, action, reward, nxtState):
        if state in self.Qtable:
            qVal = self.Qtable[state][action]
        else:
            qVal = 0

        nxtMaxQV = max(self.Qtable[nxtState].values()) if nxtState in self.Qtable else 0
        newQVal = qVal + self.alpha * (reward + self.gamma * nxtMaxQV - qVal)

        if state not in self.Qtable:
            self.Qtable[state] = {}
        self.Qtable[state][action] = newQVal

        # increment action count
        self.action_count[state][action] += 1

    # decide the action based on UCB
    def decideAction(self, trfLight):
        if self.currentCooldown > 0:
            self.currentCooldown -= 1
            return trfLight.color

        sensorData = SensorData(self.environment).collectData()
        state = self.getState(trfLight, sensorData)
        action = self.getAction(state)

        self.environment.updateEnvironment()  # Update the environment
        nextState = self.getState(trfLight, sensorData)
        reward = self.getReward(trfLight, simulation_time=None)

        # update the Q-table with the new reward
        self.updateQTable(state, action, reward, nextState)
        
        self.currentCooldown = self.decCoolDown
        return action

    def getReward(self, trfLight, simulation_time = None):
        if simulation_time is None:
            simulation_time = datetime.datetime.now()

        # look at the relativ positions and numbers of pds and vehicles that have passed the traffic light
        numVehiclePassed = sum(1 for vehicle in self.environment.vehicles if vehicle.x == trfLight.x and vehicle.y == trfLight.y)
        #count vehicles waiting directly before the traffic light
        numVehiclesWaiting = sum(1 for vehicle in self.environment.vehicles if vehicle.x == trfLight.x - 1 and vehicle.y == trfLight.y)
        #count thr vrhivles that are back from the traffic light() more than 1 unit behind
        numVehiclesFAR = sum(1 for vehicle in self.environment.vehicles if vehicle.x < trfLight.x - 1 and vehicle.y == trfLight.y)
        # we couount pedestrians waitiing 
        numbpdsWaiting = sum(1 for pd in self.environment.pds if pd.y == trfLight.y - 1 and pd.x == trfLight.x)
        #count the number of pedestrians waiting far from the crosswalk (further than 1 unit)
        numbPedestrinasfar = sum(1 for pd in self.environment.pds if pd.y < trfLight.y - 1 and pd.x == trfLight.x)

        # give weights to different factors( vehicles, pedestrians waiting, based on importance)
        vWeight = 0.5 #base weight for vehicles
        pWeight = 0.1 # base weight for pedestrians
        waitingWait = 0.1 #weight for vehilces waiting close to the light
        farWeight = 0.1 # and far from the light

        # Adjust penalty based on vehicle types
        vehicleScore = 0
        for vehicle in self.environment.vehicles:
            if vehicle.x == trfLight.x - 1 and vehicle.y == trfLight.y:
                vehicle_type = getattr(vehicle, 'type', 'default')  # Get vehicle type or default
                type_weight = vehicle_type_weights.get(vehicle_type, vehicle_type_weights['default'])
                vehicleScore += waitingWait * type_weight  # Apply weight to the waiting penalty


        #dynamic weights adjustment for rush hour
    
        for vehicle in self.environment.vehicles:
            if vehicle.x == trfLight.x - 1 and vehicle.y == trfLight.y:
                if vehicle not in self.vehicle_arrival_times:
                    self.vehicle_arrival_times[vehicle] = simulation_time

        is_rush_hour = 7 <= simulation_time.hour <= 9 or 17 <= simulation_time.hour <= 19
        traffic_density = numVehiclesWaiting + numVehiclesFAR + numbpdsWaiting + numbPedestrinasfar
        vWeight, pWeight = self.get_dynamic_weights( is_rush_hour, traffic_density)

        waitingWait = 0.1
        farWeight = 0.1

        # the weighted sum of different factors
        # penalize vehicles waiting and those far away by applying the weights
        vehicleScore = (numVehiclesWaiting * waitingWait + numVehiclesFAR * farWeight) * vWeight
        pedestianScore = (numbpdsWaiting * waitingWait + numbPedestrinasfar * farWeight) * pWeight

        # raward for letting vehicles pass and penalize for waiting pds
        # higher score if more vehicles pass and fewer penalties for waiting vehicles/pedestrians
        reward = numVehiclePassed - vehicleScore - pedestianScore

        current_waiting_vehicles = [v for v in self.environment.vehicles if v.x == trfLight.x - 1 and v.y == trfLight.y]
        vehicle_wait_times = self.calculate_vehicle_wait_times(current_waiting_vehicles, simulation_time)


        # New penalties
        # wait time penalty
        max_vehicles_wait = 30  # Maximum wait time for vehicles in seconds
        max_pedestrian_wait = 10  # Maximum wait time for pedestrians in seconds
        wait_time_penalty = self.calculate_wait_time_penalty(
            vehicle_wait_times,
            [p for p in self.environment.pds if p.y == trfLight.y - 1 and p.x == trfLight.x],
            max_vehicles_wait,
            max_pedestrian_wait
        )
        reward -= wait_time_penalty

        # safety penalty for pedestrians
        max_safe_wait_time = 15  # Maximum safe wait time for pedestrians in seconds
        pedestrians_waiting = [p for p in self.environment.pds if p.y == trfLight.y - 1 and p.x == trfLight.x]
        safety_penalty = self.calculate_safety_penalty(pedestrians_waiting, max_safe_wait_time)
        reward -= safety_penalty

        # congestion penalty
        congestion_threshold = 10  # Number of vehicles/pedestrians that indicate congestion
        congestion_penalty = max(0, (numVehiclesWaiting + numbpdsWaiting - congestion_threshold) * 0.5)
        reward -= congestion_penalty

        # frequent switching penalty
        if hasattr(self, 'last_action') and self.last_action != trfLight.color:
            switching_penalty = 2
            reward -= switching_penalty
        self.last_action = trfLight.color

        self.clean_up_vehicle_arrival_times(trfLight)

        return reward

    def clean_up_vehicle_arrival_times(self, trfLight):
        # Remove vehicles that have passed the traffic light from vehicle_arrival_times
        self.vehicle_arrival_times = {v: t for v, t in self.vehicle_arrival_times.items() 
                                      if not (v.x == trfLight.x and v.y == trfLight.y)}


    # pedestrian and vehivle wait time adjustments
    def calculate_wait_time_penalty(self, vehicle_wait_times, pedestrians_waiting, max_vehicles_wait, max_pedestrian_wait):
        vehicle_wait_penalty = sum(min(wait_time, max_vehicles_wait) for wait_time in vehicle_wait_times)
        pedestrian_wait_penalty = sum(min(getattr(pedestrian, 'wait_time', 0), max_pedestrian_wait) for pedestrian in pedestrians_waiting)
        
        pedestrian_penalty_weight = 0.7
        vehicle_penalty_weight = 0.3
        
        return (vehicle_penalty_weight * vehicle_wait_penalty) + (pedestrian_penalty_weight * pedestrian_wait_penalty)

    def calculate_vehicle_wait_times(self, current_waiting_vehicles, current_time):
        wait_times = []
        for vehicle in current_waiting_vehicles:
            if vehicle in self.vehicle_arrival_times:
                wait_time = (current_time - self.vehicle_arrival_times[vehicle]).total_seconds()
                wait_times.append(wait_time)
        return wait_times

    #DYNAMIC WEIGHTS ASSIGNMENT BASED ON TRAFFIC CONDITIONS
    def get_dynamic_weights(self, is_rush_hour, traffic_density):
        #check if it is rush hour or if traffic density is above a certain threshold
        if is_rush_hour or traffic_density > 20:
           #prioritize vehicles during rush hour by giveing them a higher weight to make traffic flow more smoothly
            vehicle_weight = 1.5 #increase importance of
            pedestrian_weight = 1.0

        else:
            #if it is not rush  hour, prioritize the pedestrians
            vehicle_weight = 1.0
            pedestrian_weight = 1.5

        return vehicle_weight, pedestrian_weight
    

    # SAFETY EMPHASIS FOR PEDESTRIAN WAITING TIMES
    def calculate_safety_penalty(self,pedestrians_waiting, max_safe_wait_time):
        # we initialize the safety_penalty at time 0
        safety_penalty = 0

        for pedestrian in pedestrians_waiting:
            #we iterate over each pedestrian that is waiting
            #if a pedestrians waiting time exceeds the safe limit, we apply a heavy pen
            if pedestrian.wait_time > max_safe_wait_time:
                #add a penalty for every second over the safe wait time
                safety_penalty += (pedestrian.wait_time - max_safe_wait_time) * 2

        return safety_penalty

class AIControllerBoltmann:
    def __init__(self, trfLightSystem, environment, alpha=0.7, gamma=0.1, tau=0.7):
        self.trfLightSystem = trfLightSystem
        self.environment = environment
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.tau = tau  # temperature parameter for softmax exploration
        self.Qtable = {}  # Q table for action values
        self.decCoolDown = 10  # Minimum frames between decisions
        self.currentCooldown = 0
        self.vehicle_arrival_times = {}

    # Get the current state based on traffic and pedestrian data
    def getState(self, trfLight, sensorData):
        objDetector = ObjectDetector(sensorData)
        decVehicls = objDetector.detectVehicles()
        decPeds = objDetector.detectpds()

        numVehicles = sum(1 for vehicle in decVehicls if vehicle[0] == trfLight.x - 1 and vehicle[1] == trfLight.y)
        numbPedestrinas = sum(1 for pd in decPeds if pd[1] == trfLight.y - 1 and pd[0] == trfLight.x)
        state = (numVehicles, numbPedestrinas)
        return state

    def getAction(self, state):
        # Ensure the state exists in the Q table
        if state not in self.Qtable:
            self.Qtable[state] = {'green': 0, 'red': 0}  # Initialize Q-values for both actions
        
        # Get Q-values for all actions
        q_values = np.array(list(self.Qtable[state].values())) 
        # Apply the softmax numerator (e^(Q / tau))
        exp_values = np.exp(q_values / self.tau)  
        # Normalize to get probabilities
        #the boltzman function
        probabilities = exp_values / np.sum(exp_values) 

        # Choose an action based on the computed probabilities
        actions = list(self.Qtable[state].keys())
        action = np.random.choice(actions, p=probabilities)
        return action

    # Update Q table after taking an action
    def updateQTable(self, state, action, reward, nxtState):
        if state in self.Qtable:
            qVal = self.Qtable[state][action]
        else:
            qVal = 0

        nxtMaxQV = max(self.Qtable[nxtState].values()) if nxtState in self.Qtable else 0
        newQVal = qVal + self.alpha * (reward + self.gamma * nxtMaxQV - qVal)

        if state not in self.Qtable:
            self.Qtable[state] = {}
        self.Qtable[state][action] = newQVal

    # decide the action based on Boltzmann exploration
    def decideAction(self, trfLight):
        if self.currentCooldown > 0:
            self.currentCooldown -= 1
            return trfLight.color

        sensorData = SensorData(self.environment).collectData()
        state = self.getState(trfLight, sensorData)
        action = self.getAction(state)

        self.environment.updateEnvironment()  # Update the environment
        nextState = self.getState(trfLight, sensorData)
        reward = self.getReward(trfLight, simulation_time=None)

        # Update the Q-table with the new reward
        self.updateQTable(state, action, reward, nextState)
        
        self.currentCooldown = self.decCoolDown
        return action

    #same as the previous ones
    def getReward(self, trfLight, simulation_time = None):
        if simulation_time is None:
            simulation_time = datetime.datetime.now()

        # look at the relativ positions and numbers of pds and vehicles that have passed the traffic light
        numVehiclePassed = sum(1 for vehicle in self.environment.vehicles if vehicle.x == trfLight.x and vehicle.y == trfLight.y)
        #count vehicles waiting directly before the traffic light
        numVehiclesWaiting = sum(1 for vehicle in self.environment.vehicles if vehicle.x == trfLight.x - 1 and vehicle.y == trfLight.y)
        #count thr vrhivles that are back from the traffic light() more than 1 unit behind
        numVehiclesFAR = sum(1 for vehicle in self.environment.vehicles if vehicle.x < trfLight.x - 1 and vehicle.y == trfLight.y)
        # we couount pedestrians waitiing 
        numbpdsWaiting = sum(1 for pd in self.environment.pds if pd.y == trfLight.y - 1 and pd.x == trfLight.x)
        #count the number of pedestrians waiting far from the crosswalk (further than 1 unit)
        numbPedestrinasfar = sum(1 for pd in self.environment.pds if pd.y < trfLight.y - 1 and pd.x == trfLight.x)

        # give weights to different factors( vehicles, pedestrians waiting, based on importance)
        vWeight = 0.5 #base weight for vehicles
        pWeight = 0.1 # base weight for pedestrians
        waitingWait = 0.6 #weight for vehilces waiting close to the light
        farWeight = 0.05 # and far from the light

        # Adjust penalty based on vehicle types
        vehicleScore = 0
        for vehicle in self.environment.vehicles:
            if vehicle.x == trfLight.x - 1 and vehicle.y == trfLight.y:
                vehicle_type = getattr(vehicle, 'type', 'default')  # Get vehicle type or default
                type_weight = vehicle_type_weights.get(vehicle_type, vehicle_type_weights['default'])
                vehicleScore += waitingWait * type_weight  # Apply weight to the waiting penalty


        #dynamic weights adjustment for rush hour
    
        for vehicle in self.environment.vehicles:
            if vehicle.x == trfLight.x - 1 and vehicle.y == trfLight.y:
                if vehicle not in self.vehicle_arrival_times:
                    self.vehicle_arrival_times[vehicle] = simulation_time

        is_rush_hour = 7 <= simulation_time.hour <= 9 or 17 <= simulation_time.hour <= 19
        traffic_density = numVehiclesWaiting + numVehiclesFAR + numbpdsWaiting + numbPedestrinasfar
        vWeight, pWeight = self.get_dynamic_weights( is_rush_hour, traffic_density)

        waitingWait = 5
        farWeight = 0.1

        # the weighted sum of different factors
        # penalize vehicles waiting and those far away by applying the weights
        vehicleScore = (numVehiclesWaiting * waitingWait + numVehiclesFAR * farWeight) * vWeight
        pedestianScore = (numbpdsWaiting * waitingWait + numbPedestrinasfar * farWeight) * pWeight

        # raward for letting vehicles pass and penalize for waiting pds
        # higher score if more vehicles pass and fewer penalties for waiting vehicles/pedestrians
        reward = numVehiclePassed - 10*vehicleScore - pedestianScore

        current_waiting_vehicles = [v for v in self.environment.vehicles if v.x == trfLight.x - 1 and v.y == trfLight.y]
        vehicle_wait_times = self.calculate_vehicle_wait_times(current_waiting_vehicles, simulation_time)


        # New penalties
        # 1. Wait time penalty
        max_vehicles_wait = 30  # Maximum wait time for vehicles in seconds
        max_pedestrian_wait = 10  # Maximum wait time for pedestrians in seconds
        wait_time_penalty = self.calculate_wait_time_penalty(
            vehicle_wait_times,
            [p for p in self.environment.pds if p.y == trfLight.y - 1 and p.x == trfLight.x],
            max_vehicles_wait,
            max_pedestrian_wait
        )
        reward -= wait_time_penalty

        # safety penalty for pedestrians
        max_safe_wait_time = 15  # Maximum safe wait time for pedestrians in seconds
        pedestrians_waiting = [p for p in self.environment.pds if p.y == trfLight.y - 1 and p.x == trfLight.x]
        safety_penalty = self.calculate_safety_penalty(pedestrians_waiting, max_safe_wait_time)
        reward -= safety_penalty

        # congestion penalty
        congestion_threshold = 10  # Number of vehicles/pedestrians that indicate congestion
        congestion_penalty = max(0, (numVehiclesWaiting + numbpdsWaiting - congestion_threshold) * 0.5)
        reward -= congestion_penalty

        # frequent switching penalty
        if hasattr(self, 'last_action') and self.last_action != trfLight.color:
            switching_penalty = 2
            reward -= switching_penalty
        self.last_action = trfLight.color

        self.clean_up_vehicle_arrival_times(trfLight)

        return reward

    def clean_up_vehicle_arrival_times(self, trfLight):
        # Remove vehicles that have passed the traffic light from vehicle_arrival_times
        self.vehicle_arrival_times = {v: t for v, t in self.vehicle_arrival_times.items() 
                                      if not (v.x == trfLight.x and v.y == trfLight.y)}


    # pedestrian and vehivle wait time adjustments
    def calculate_wait_time_penalty(self, vehicle_wait_times, pedestrians_waiting, max_vehicles_wait, max_pedestrian_wait):
        vehicle_wait_penalty = sum(min(wait_time, max_vehicles_wait) for wait_time in vehicle_wait_times)
        pedestrian_wait_penalty = sum(min(getattr(pedestrian, 'wait_time', 0), max_pedestrian_wait) for pedestrian in pedestrians_waiting)
        
        pedestrian_penalty_weight = 0.7
        vehicle_penalty_weight = 0.3
        
        return (vehicle_penalty_weight * vehicle_wait_penalty) + (pedestrian_penalty_weight * pedestrian_wait_penalty)

    def calculate_vehicle_wait_times(self, current_waiting_vehicles, current_time):
        wait_times = []
        for vehicle in current_waiting_vehicles:
            if vehicle in self.vehicle_arrival_times:
                wait_time = (current_time - self.vehicle_arrival_times[vehicle]).total_seconds()
                wait_times.append(wait_time)
        return wait_times

    #DYNAMIC WEIGHTS ASSIGNMENT BASED ON TRAFFIC CONDITIONS
    def get_dynamic_weights(self, is_rush_hour, traffic_density):
        #check if it is rush hour or if traffic density is above a certain threshold
        if is_rush_hour or traffic_density > 20:
           #prioritize vehicles during rush hour by giveing them a higher weight to make traffic flow more smoothly
            vehicle_weight = 1.5 #increase importance of
            pedestrian_weight = 1.0

        else:
            #if it is not rush hour, prioritize the pedestrians
            vehicle_weight = 1.0
            pedestrian_weight = 1.5

        return vehicle_weight, pedestrian_weight
    

    # SAFETY EMPHASIS FOR PEDESTRIAN WAITING TIMES
    def calculate_safety_penalty(self,pedestrians_waiting, max_safe_wait_time):
        # we initialize the safety_penalty at time 0
        safety_penalty = 0

        for pedestrian in pedestrians_waiting:
            #we iterate over each pedestrian that is waiting
            #if a pedestrians waiting time exceeds the safe limit, we apply a heavy pen
            if pedestrian.wait_time > max_safe_wait_time:
                #add a penalty for every second over the safe wait time
                safety_penalty += (pedestrian.wait_time - max_safe_wait_time) * 2

        return safety_penalty

    
#DeterministicController controls traffic based on predefined time interval
class DeterministicController:
    def __init__(self, trfLightSystem, environment, interval = 20):
        self.trfLightSystem = trfLightSystem
        self.environment = environment
        self.interval = interval  #time interval in seconds
        self.last_switch_time = datetime.datetime.now()


    #choses what to do next after getting the actions and the states
    def decideAction(self, trfLight):
        currentTime = datetime.datetime.now()
        timePassed = (currentTime - self.last_switch_time).total_seconds()

        if timePassed >= self.interval:
            if trfLight.color == 'red':
                action = 'green'
            else:
                action = 'red'
            
            self.last_switch_time = currentTime
        
        else:
            action = trfLight.color

        return action





class Vehicle:
    def __init__(self, id, type, x, y, environment, root):
        self.id = id
        self.type = type
        self.x = x
        self.y = y
        self.environment = environment  # save the Environment instance
        self.root = root
        self.waiting = False
        self.speed = 1

        # Set color based on vehicle type
        if self.type == 'car':
            self.color = "blue"
        elif self.type == 'bike':
            self.color = "orange"
        elif self.type == 'bus':
            self.color = "pink"
        elif self.type == 'truck':
            self.color = "cyan"
        else:
            self.color = "gray"
    
    def xIncrease(self):
        self.x += 1
        
    def is_waiting(self):
        return self.waiting

    def move(self, grid_width, trfLight, root):
        old_x = self.x
        if 0 <= self.x < grid_width - 1:
            # see if the vehicle is at the traffic light
            if (self.x + 1 == trfLight.x and self.y == trfLight.y) or (self.x + 1 == trfLight.x and self.y -1 == trfLight.y):
                # look and see if the traffic light is green and the cell in front is empty
                if trfLight.color == 'green' and not any(v.x == self.x + 1 and v.y == self.y for v in self.environment.vehicles):
                    #because we dont want them to bump into each other
                    #root.after(1000, self.xIncrease)
                    self.x = trfLight.x+1
                    self.x += 1
                else:
                    pass
            else:
                # take out the vehicle forward if the cell in front is empty
                if not any(v.x == self.x + 1 and v.y == self.y for v in self.environment.vehicles):
                    #root.after(100, self.xIncrease())
                    self.x += 1

        
        else:
            # take out the vehicle if it is out of bounds
            self.environment.vehicles.remove(self)
            #self.environment.grid[self.y][self.x] = None
        
        self.waiting = ((self.x  < trfLight.x) and (self.y == trfLight.y or self.y-1 == trfLight.y) and trfLight.color != 'green')
        self.speed = self.x - old_x


class Pedestrian:
    def __init__(self, id, x, y, environment):
        self.id = id
        self.x = x
        self.y = y
        self.environment = environment  # store the Environment instance
        self.wait_time = 0
        self.waiting = False
        self.speed = 1

    def is_waiting(self):
        return self.waiting

    def walk(self, grid_height, trfLight):
        old_y = self.y
        # Increment the wait time if the pedestrian is waiting at a red light
        if (self.y + 1 == trfLight.y and self.x == trfLight.x) or (self.y + 1 == trfLight.y and self.x - 1 == trfLight.x):
            if trfLight.color == 'red':
                self.wait_time += 1  # increase wait time when the pedestrian is blocked
            else:
                #self.wait_time = 0  # also Reset waiting time if they are moving
                pass

        if 0 <= self.y < grid_height - 1:
            # see if the pd is at the traffic light
            #obj = self.environment.grid[self.y+1][self.x]
            if (self.y + 1 == trfLight.y and self.x == trfLight.x) or (self.y + 1 == trfLight.y and self.x -1 == trfLight.x):
                # see if the traffic light is red and the cell in front is empty
                if trfLight.color == 'red':# 
                    self.y = trfLight.x+1
                    self.y += 1
                else:
                    pass
            else:
                # move the pd forward if the cell in front is empty
                if not any(p.x == self.x and p.y == self.y + 1 for p in self.environment.pds):
                    self.y += 1
                else:
                    pass
        else:
            # remove the pd if it is out of bounds
            self.environment.pds.remove(self)
            self.environment.grid[self.y][self.x] = None

        self.waiting = (self.y  < trfLight.y and (self.x == trfLight.x or self.x - 1 == trfLight.x)  and trfLight.color != 'red')
        self.speed = self.y - old_y




class Environment:
    def __init__(self, root, width, height):
        self.root = root  # stores the root instance
        self.width = width
        self.height = height
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        #stores the data read from the dataset
        self.vehicles = []
        self.pds = []
        self.trafficLights = []
        self.sensors = []
        #used to control the the update frequency for vehicles and pedestrians
        self.updateCounter = 0
        self.vehicleUpdateFrequency = 4  # Updates vehicles every frame
        self.pedestrianUpdateFrequency = 30  # Update pedestrians every 30 frames
        #but these values will be updated with thw gui  also
        self.simulation_start_time = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        self.simulation_time = df['Time'].min()  # start with the earliest time in the dataset
        self.vehicle_queue = []  # Queue to hold vehicles to be added
        self.pedestrian_queue = [] #same for pedestrians 

        self.current_time = 0

        self.performance_metrics = PerformanceMetrics()


    def addObject(self, obj):
        #add it here for better stats calculations
        if isinstance(obj, Vehicle):
            self.vehicles.append(obj)
        elif isinstance(obj, Pedestrian):
            self.pds.append(obj)
        elif isinstance(obj, TrafficLight):
            self.trafficLights.append(obj)
        
        #add the element on to the grid
        self.grid[obj.y][obj.x] = obj

    #actual calling of the add objects
    def addNewOBject(self, objType):
        if not self.vehicle_queue and not self.pedestrian_queue:
            traffic_data = getTrafficData(self.simulation_time)
            self.simulation_time = traffic_data['time']
            
            # Add different types  vehicles to the queue, like bikes buses and stuff
            for _ in range(traffic_data['cars']):
                self.vehicle_queue.append(('car', random.choice([9,10])))
            for _ in range(traffic_data['bikes']):
                self.vehicle_queue.append(('bike', random.choice([9,10])))
            for _ in range(traffic_data['buses']):
                self.vehicle_queue.append(('bus', random.choice([9,10])))
            for _ in range(traffic_data['trucks']):
                self.vehicle_queue.append(('truck', random.choice([9,10])))
            
            # Add pedestrians to the queue also
            for _ in range(traffic_data['pedestrians']):
                self.pedestrian_queue.append(random.choice([9,10]))

        if objType == 'vehicle' and self.vehicle_queue:
            vehicle_type, y = self.vehicle_queue.pop(0)
            new_obj = Vehicle(id=f"V{len(self.vehicles)}", type=vehicle_type, x=0, y=y, environment=self, root=self.root)
            self.addObject(new_obj)
        
        elif objType == 'pd' and self.pedestrian_queue:
            x = self.pedestrian_queue.pop(0)
            new_obj = Pedestrian(id=f"P{len(self.pds)}", x=x, y=0, environment=self)
            self.addObject(new_obj)

        #interval at which vehicles are shown
        next_time = 700  # default to 700milisecondss
        if self.vehicle_queue:
            next_time = 700  # If there are vehicles in the queue, add them faster
        self.root.after(next_time, self.addNewOBject, objType)

    def addObjects(self):
        #another attempt to add multiple traffic light systems
        self.addNewOBject('vehicle')
        self.addNewOBject('pd')
        #rounds to add objects

    def updateEnvironment(self):
        self.updateCounter += 1
        if self.trafficLights:
            trfLight = self.trafficLights[0]
        else:
            trfLight = None

        # update vehicles more frequenly
        if self.updateCounter % self.vehicleUpdateFrequency == 0:
            for v in self.vehicles:
                if 0 <= v.x < self.width and 0 <= v.y < self.height:
                    self.grid[v.y][v.x] = None  # remove old position
                if trfLight:
                    v.move(self.width, trfLight, self.root)
                if 0 <= v.x < self.width and 0 <= v.y < self.height:
                    self.grid[v.y][v.x] = v  # update new position

        # update pedestrians less frequently
        if self.updateCounter % self.pedestrianUpdateFrequency == 0:
            for pd in self.pds:
                if 0 <= pd.x < self.width and 0 <= pd.y < self.height:
                    self.grid[pd.y][pd.x] = None  # remove old position
                if trfLight:
                    pd.walk(self.height, trfLight)
                if 0 <= pd.x < self.width and 0 <= pd.y < self.height:
                    self.grid[pd.y][pd.x] = pd  # update new position

        for light in self.trafficLights:
            self.grid[light.y][light.x] = light  # ensure traffic light is not removed
    

        self.performance_metrics.update(self)

# we can calculate the density of vehicles and pedestrians in specific areas of the grid
# and use heatmap - like color variations to visualize desnsity 
# Areas ith high traffic (vehicles/pedestrian) will be have a grey circle around
#  



class TrafficSimulationGUI:
    def __init__(self, root, environment):
        self.root = root
        self.environment = environment
        #boolean to control pause/resume 
        self.paused = False
        #create a main frame tk variable
        self.main_frame = tk.Frame(root)
        #start packing things on it
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        #create canvas
        self.canvas = tk.Canvas(self.main_frame, width=600, height=600, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.control_panel = tk.Frame(self.main_frame, width=200)
        self.control_panel.pack(side=tk.RIGHT, fill=tk.Y)
        self.create_control_panel()
        self.draw_grid()

        self.metrics_label = tk.Label(self.control_panel, text="", justify=tk.LEFT)
        self.metrics_label.pack(pady=10)
    #control some of the parameters
    def create_control_panel(self):
        self.pause_button = tk.Button(self.control_panel, text="Pause", command=self.toggle_pause)
        self.pause_button.pack(pady=10)
        
        tk.Label(self.control_panel, text="Vehicle Update Frequency").pack()
        self.vehicleFreqSlider = tk.Scale(self.control_panel, from_=1, to=10, orient=tk.HORIZONTAL, command=self.update_vehicle_freq)
        self.vehicleFreqSlider.set(self.environment.vehicleUpdateFrequency)
        self.vehicleFreqSlider.pack()

        tk.Label(self.control_panel, text="Pedestrian Update Frequency").pack()
        self.pedFreqSlider = tk.Scale(self.control_panel, from_=10, to=50, orient=tk.HORIZONTAL, command=self.update_ped_freq)
        self.pedFreqSlider.set(self.environment.pedestrianUpdateFrequency)
        self.pedFreqSlider.pack()

        # reward function weight weight sliders
        tk.Label(self.control_panel, text = "Vehicle Wait Weight").pack()
        self.vehicleWeightSlider = tk.Scale(self.control_panel, from_=0, to = 1, resolution=0.1, orient=tk.HORIZONTAL, command=self.update_vehicle_weight)
        self.vehicleWeightSlider.set(0.5)
        self.vehicleWeightSlider.pack()

        tk.Label(self.control_panel, text="Pedestrian Wait Weight").pack()
        self.pedWeightSlider = tk.Scale(self.control_panel, from_=0, to=1, resolution=0.1, orient=tk.HORIZONTAL, command=self.update_pad_weight)
        self.pedWeightSlider.set(0.1)
        self.pedWeightSlider.pack()

        tk.Label(self.control_panel, text="Pedestrian = Purple").pack()
        tk.Label(self.control_panel, text="Vehicles = Others").pack()

    def update_vehicle_weight(self, value):
        self.environment.vehicle_weight = float(value)
    
    def update_pad_weight(self, value):
        self.environment.pedestrian_weight = float(value)

    def draw_grid(self):
        _width = 30
        _height = 30
        for y in range(self.environment.height):
            for x in range(self.environment.width):
                x0 = x * _width
                y0 = y * _height
                x1 = x0 + _width
                y1 = y0 + _height
                self.canvas.create_rectangle(x0, y0, x1, y1, outline="black")
#toggle pause and resume button handling
    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_button.config(text="Resume")
        else:
            self.pause_button.config(text="Pause")
            self.updateDisplay()

    def update_vehicle_freq(self, value):
        self.environment.vehicleUpdateFrequency = int(value)

    def update_ped_freq(self, value):
        self.environment.pedestrianUpdateFrequency = int(value)

    def updateDisplay(self):
        if not self.paused:
            #clear canvas
            self.canvas.delete("all")
            self.draw_grid()
            #self.draw_density_overlays()
            for y in range(self.environment.height):
                for x in range(self.environment.width):
                    obj = self.environment.grid[y][x]
                    if obj is not None:
                        if isinstance(obj, TrafficLight):
                            self.drawLight(x, y, obj.color)
                        elif isinstance(obj, Vehicle):
                            color = obj.color
                            self.drawObject(x, y, color)
                        elif isinstance(obj, Pedestrian):
                            color = "purple"
                            self.drawObject(x, y, color)
            self.environment.updateEnvironment()
            self.update_metrics()
            self.root.after(50, self.updateDisplay)

    def update_metrics(self):
        metrics_text = self.environment.performance_metrics.report()
        self.metrics_label.config(text=metrics_text)

    def drawObject(self, x, y, color):
        _width = 30
        _height = 30
        x0 = x * _width + 5
        y0 = y * _height + 5
        x1 = x0 + _width - 10
        y1 = y0 + _height - 10
        self.canvas.create_rectangle(x0, y0, x1, y1, fill=color)

    def drawLight(self, x, y, color):
        _width = 60
        _height = 60
        x0 = x*_width - 270
        y0 = y*_height - 270
        x1 = x0 + _width -0
        y1 = y0 + _height - 0
        self.canvas.create_rectangle(x0, y0, x1, y1, fill=color)


class PerformanceMetrics:
    def __init__(self):
        #defines the metrics
        self.vehicles_waiting = 0
        self.pedestrians_waiting = 0
        self.total_vehicles_passed = 0
        self.total_pedestrians_passed = 0
        self.vehicles_passed_after_stopping = 0
        self.pedestrians_passed_after_stopping = 0
        self.total_vehicle_wait_time = 0
        self.total_pedestrian_wait_time = 0
        self.last_vehicle_count = 0
        self.last_pedestrian_count = 0
        self.stopped_vehicles = set()
        self.stopped_pedestrians = set()

        self.start_time = datetime.datetime.now()
        self.last_snapshot_time = datetime.datetime.now()

        # List to store snapshots of statistics
        self.snapshots = []

    def update(self, environment):
        traffic_light = environment.trafficLights[0]  # Assuming there's at least one traffic light
        
        # Count waiting (stopped) vehicles and pedestrians
        new_stopped_vehicles = set()
        new_stopped_pedestrians = set()
        
        
        ##
        ## 
        # Calculates and continously updates metrics 

        for v in environment.vehicles:
            #if vehicle is waiting, increase its waiting time
            if v.x < traffic_light.x and ( v.y == traffic_light.y or v.y == traffic_light.y + 1) :#and v.speed == 0:
                new_stopped_vehicles.add(v.id)
                if v.id not in self.stopped_vehicles:
                    #only new vehicles waiting times get updated
                    self.total_vehicle_wait_time += 1
        
        for p in environment.pds:
            #if pedestrian is waiting, increase ists waiting time
            if p.y < traffic_light.y and (p.x == traffic_light.x or p.x == traffic_light.x +1) and p.speed == 0:
                new_stopped_pedestrians.add(p.id)
                if p.id not in self.stopped_pedestrians:
                    self.total_pedestrian_wait_time += 1

        self.vehicles_waiting = len(new_stopped_vehicles)
        self.pedestrians_waiting = len(new_stopped_pedestrians)

        # count passed vehicles and pedestrians
        current_vehicle_count = len(environment.vehicles)
        current_pedestrian_count = len(environment.pds)

        #by comparing last counts vs the current counts

        vehicles_passed = max(0, self.last_vehicle_count - current_vehicle_count)
        pedestrians_passed = max(0, self.last_pedestrian_count - current_pedestrian_count)

        #add that onto the number of vehicles passed
        self.total_vehicles_passed += vehicles_passed
        self.total_pedestrians_passed += pedestrians_passed

        # count passed after stopping
        # (self.stopped_vehicles - new_stopped_vehicles) counts 
        #vehicles that were stopped during the previous state
        #but are no longer stopped in the current state
        self.vehicles_passed_after_stopping += len(self.stopped_vehicles - new_stopped_vehicles)
        self.pedestrians_passed_after_stopping += len(self.stopped_pedestrians - new_stopped_pedestrians)

        #update the last vehicle count, pedestrian, 
        #stopped vehicles and pedestrians
        self.last_vehicle_count = current_vehicle_count
        self.last_pedestrian_count = current_pedestrian_count
        self.stopped_vehicles = new_stopped_vehicles
        self.stopped_pedestrians = new_stopped_pedestrians

        # Check if 60 seconds have passed since the last snapshot
        if (datetime.datetime.now() - self.last_snapshot_time).seconds >= 60:
            self.take_snapshot()

    def take_snapshot(self):
        # Take a snapshot of current metrics and store it
        timeDiff = (datetime.datetime.now() - self.start_time).seconds
        snapshot = {
            'time': timeDiff,
            'vehicles_waiting': self.vehicles_waiting,
            'pedestrians_waiting': self.pedestrians_waiting,
            'total_vehicles_passed': self.total_vehicles_passed,
            'total_pedestrians_passed': self.total_pedestrians_passed,
            'vehicles_passed_after_stopping': self.vehicles_passed_after_stopping,
            'pedestrians_passed_after_stopping': self.pedestrians_passed_after_stopping,
            'total_vehicle_wait_time': self.total_vehicle_wait_time,
            'total_pedestrian_wait_time': self.total_pedestrian_wait_time
        }
        self.snapshots.append(snapshot)

        print(snapshot)

        self.last_snapshot_time = datetime.datetime.now()

    def report(self):
        avg_vehicle_wait = self.total_vehicle_wait_time / self.total_vehicles_passed if self.total_vehicles_passed > 0 else 0
        avg_pedestrian_wait = self.total_pedestrian_wait_time / self.total_pedestrians_passed if self.total_pedestrians_passed > 0 else 0
        currentTime2 = datetime.datetime.now()

        timeDiff = (currentTime2 - self.start_time).seconds
        return f"""
            Time Passed: {timeDiff}
            Performance Metrics:
            Vehicles currently waiting: {self.vehicles_waiting}
            Pedestrians currently waiting: {self.pedestrians_waiting}
            Total vehicles passed: {self.total_vehicles_passed}
            Total pedestrians passed: {self.total_pedestrians_passed}
            Vehicles passed after stopping: {self.vehicles_passed_after_stopping}
            Pedestrians passed after stopping: {self.pedestrians_passed_after_stopping}
            Average vehicle wait time: {avg_vehicle_wait:.2f}
            Average pedestrian wait time: {avg_pedestrian_wait:.2f}
                    """


def main():
    root2 = tk.Tk()
    root2.title("AI Traffic Control System")
    
    env = Environment(root=root2, width=20, height=20)
    #ADDs vehicles and pedestrians
    env.addObjects()
    # create and add some traffic lights
    tl1 = TrafficLight(id="TL1", color="red", x=9, y=9, status="active")
    env.addObject(tl1)
    tl2 = TrafficLight(id="TL2", color=tl1.color, x=10, y=10, status="active")
   
    # make the traffic light system
    tls = TrafficLightSystem(root=root2, location="Main Intersection", environment=env)
    tls.addTrafficLight(tl1)

    tls2 = TrafficLightSystem(root=root2, location="Secondary Intersection", environment=env)
    tls2.addTrafficLight(tl2)

    gui = TrafficSimulationGUI(root2, env)
    gui.update_metrics()
    tls.changeTrafficLight()

    gui.updateDisplay()

    root2.mainloop()

if __name__ == "__main__":
    main()
