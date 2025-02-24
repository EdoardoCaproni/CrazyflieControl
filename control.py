import time
import math

import queue
import threading
import readchar
import scipy as sp
import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np

import cflib.crtp
from cflib.utils import uri_helper
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.positioning.position_hl_commander import PositionHlCommander
from cflib.positioning.motion_commander import MotionCommander

from PID import PID
from PID import DefaultPIDparams

"""
DUE PAROLE SUI SISTEMI DI RIFERIMENTO
Il crayzyflie utilizza un sistema di riferimento fissato sul corpo rigido di tipo ENU
(ovvero East-North-Up, che identifica in quest'ordine la direzione positiva degli assi x-y-z).
Il fronte del drone può essere identificato dalla ricetrasmittente che sporge dal frame,
mentre il retro può essere identificato dai due led di colore blu.
Visto dall'alto il sistema di riferimento è:

        ^  asse x
        |
        |
  < ----o  asse z (uscente)
  asse y

Inoltre gli angoli sono così definiti:
- roll: (φ - phi) come rotazione intorno all'asse x (positiva in senso orario) -> positivo quando inclina verso destra
- pitch: (θ - theta) come rotazione intorno all'asse y (positiva in senso antiorario) -> positivo quando il drone si impenna
- yaw: (ψ - psi) come rotazione intorno all'asse z (positiva in senso orario) - positivo virando verso sinistra

In angoli di Eulero, questo ricorda lo standard 1-2-3 (ordine di rotazione degli assi)
"""

URI = uri_helper.uri_from_env()

AUTOMATIC_CONTROL = 0
MANUAL_CONTROL = 1 #bool

MAX_UINT16 = 65535 #maximum unsigned integer (used for PWM/motor control)
HOVER_THRUST = 36000 #default value to obtain stationary behaviour in altitude (32768/48000)

MOTOR_MAX_FORCE = 0.15 #N, max force that can be exherted by 1 motor
MOTOR_MAX_TORQUE = 0.91e-3 #N*m, torque outputted at max force by 1 motor
MAX_FLIGHT_TIME = 7 #in minutes
DEFAULT_HEIGHT = 1 #in meters

T = 0.01 #s, step time, must be >= CF logging rate (0.01) + the cycle computation time

DRONE_DIAGONAL_LENGHT = 0.092 #in m; distance between opposite motors
DRONE_SIDE_LENGHT = 0.092/math.sqrt(2) #in m; distance between adiacent motors
DRONE_MASS = 28.0e-3 #Kg, +/- 0.1 grams
DRONE_FORCE_TO_TORQUE = 0.00596 #proportional factor, between the force of a motor and the torque it generates

user_input = queue.Queue(1) #a queue with maxsize=1 is used to ensure that operations between threads are atomic


class Sensor_logging:
    battery_level = 0

    origin = [0, 0, 0] #position offset (in m) so that (x, y, z) are relative to it. Set with set_origin()
    x, y, z = 0, 0, 0 #position in mm
    vx, vy, vz = 0, 0, 0 #velocity in mm/s
    #ax, ay, az = 0, 0, 0 #acceleration in mm/s^2, Z includes gravity

    qw, qx, qy, qz = 0, 0, 0, 0 #attitude as a quaternion, unpacked from raw packet sent by logging
    roll, pitch, yaw = 0, 0, 0 #euler angles in degrees

    p, q, r = 0, 0, 0 #angular velocity in millirad/s
    roll_rate, pitch_rate, yaw_rate = 0, 0, 0 #euler angle rates in millirad/s

    def set_origin(self): #ONLY CALL ONCE!
        self.origin[0] = self.x
        self.origin[1] = self.y
        self.origin[2] = self.z

    ### CALLBACKS ###
    def battery_log_cb (self, timestamp, data, logconf):
        self.battery_level=data["pm.batteryLevel"]
        
    def position_cb(self, timestamp, data, logconf):
        #the starting position (origin) of the crazyflie is subtracted from x, y, z to provide the distance relative to it
        self.x = data["stateEstimateZ.x"] - self.origin[0]
        self.y = data["stateEstimateZ.y"] - self.origin[1]
        self.z = data["stateEstimateZ.z"] - self.origin[2]

    def velocity_cb(self, timestamp, data, logconf):
        self.vx = data["stateEstimateZ.vx"]
        self.vy = data["stateEstimateZ.vy"]
        self.vz = data["stateEstimateZ.vz"]
        
    """def acceleration_cb (self, timestamp, data, logconf):
        self.ax = data["stateEstimateZ.ax"]
        self.ay = data["stateEstimateZ.ay"]
        self.az = data["stateEstimateZ.az"]"""

    def attitude_cb(self, timestamp, data, logconf):
        #unpacking algorithm from the crazyflie firmware, original in  C (modules/interface/quatcompress.h)
        comp = data["stateEstimateZ.quat"]
        mask = (1 << 9) - 1
        i_largest = comp >> 30
        sum_squares = 0
        q = [0, 0, 0, 0]

        #unpacking the quaternion
        for i in range(3, -1, -1): #3, 2, 1, 0
            if (i != i_largest):
                mag = comp & mask
                negbit = (comp >> 9 ) & 0x1
                comp = comp >> 10
                q[i] = 1/math.sqrt(2) * mag / mask
                if (negbit == 1):
                    q[i] = -q[i]
                sum_squares += q[i] * q[i]

        q[i_largest] = math.sqrt(1.0 - sum_squares)

        #update quaternion
        self.qx, self.qy, self.qz, self.qw = q

        #update attitude
        roll_r, pitch_r, yaw_r = euler_from_quaternion(*q)

        self.roll, self.pitch, self.yaw = math.degrees(roll_r), math.degrees(pitch_r), math.degrees(yaw_r)

    def angular_velocity_cb(self, timestamp, data, logconf):
        self.p = data["stateEstimateZ.rateRoll"]
        self.q = data["stateEstimateZ.ratePitch"]
        self.r = data["stateEstimateZ.rateYaw"]

        s_roll = math.sin(math.radians(self.roll))
        c_roll = math.cos(math.radians(self.roll))
        c_pitch = math.cos(math.radians(self.pitch))
        t_pitch = math.tan(math.radians(self.pitch))

        self.roll_rate = self.p - s_roll*t_pitch * self.q - c_roll*t_pitch * self.r
        self.pitch_rate = c_roll * self.q - s_roll * self.r
        self.yaw_rate = s_roll/c_pitch * self.q + c_roll/c_pitch * self.r

    ### PRINT STATES ###
    def print_battery_level(self):
        print("BATTERY LEVEL: {0}% - {1} MINUTES OF EXPECTED FLIGHT TIME".format(self.battery_level, self.battery_level*MAX_FLIGHT_TIME/100))
    def print_position(self):
        print("POSITION:\n X: {0:.1f} mm,     Y: {1:.1f} mm,     Z: {2:.1f} mm".format(self.x, self.y, self.z))    
    def print_velocity(self):
        print("VELOCITY:\n X: {0:.1f} mm/s,   Y: {1:.1f} mm/s,   Z: {2:.1f} mm/s".format(self.vx, self.vy, self.vz))  
    """def print_aceleration(self):     
        print("ACCELERATION:\n X: {0:.4f} m/s^2, Y: {1:.4f} m/s^2, Z: {2:.4f} m/s^2".format(self.ax/1000, self.ay/1000, self.az/1000))"""
    def print_attitude(self):
        print("ATTITUDE:\n ROLL: {0:.4f}°, PITCH: {1:.4f}°, YAW: {2:.4f}°".format((self.roll),
                                                                                  (self.pitch),
                                                                                  (self.yaw)))
    def print_quaternion(self):
        print("QUATERNION:\n [{0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}]".format(self.qw, self.qx, self.qy, self.qz))
    def print_rates(self):
        print("EULER RATES: \n ROLL: {0:.4f}°/s, PITCH: {1:.4f}°/S, YAW: {2:.4f}°/s".format(math.degrees(self.roll_rate/1000),
                                                                                            math.degrees(self.pitch_rate/1000),
                                                                                            math.degrees(self.yaw_rate/1000)))
    def print_angular_velocity(self):
        print("A. VELOCITY:\n P: {0:.4f}°/s, Q: {1:.4f}°/S, R: {2:.4f}°/s".format(math.degrees(self.p/1000),
                                                                                  math.degrees(self.q/1000),
                                                                                  math.degrees(self.r/1000)))
    def print_all(self):
        self.print_position()
        self.print_velocity()
        self.print_attitude()
        self.print_rates()

def clear_screen():
    print("\033[H\033[J", end="")

### MOTOR CONTROL ###
def set_motor_power (crazyflie, motor_n, *, power=0, percent=False):
    """Mask to streamline the motor PWM control
    @crazyflie is the SyncCrazyFlie instance,
    @motor_n is the selected motor (1 to 4)
    @power is the PWM value in uint_16 (from 0 to 65535), or float (0% to 100%)
    @percent is a bool used to select the @power mode"""

    if (motor_n < 1 or motor_n > 4):
        raise ValueError("Please select a motor number between 1 and 4")
    
    if not percent and type(power) != int:
        raise TypeError("Motor power must be expressed in integers when percent flag is False")

    if not percent and (power<0 or power>MAX_UINT16):
        raise ValueError("Motor power must be set between 0 and {0}".format(MAX_UINT16))
    
    if percent and (power<0 or power>100):
        raise ValueError("Motor power must be set between 0%% and 100%")
    
    if percent:
        power = int(MAX_UINT16 * power / 100)

    crazyflie.cf.param.set_value("motorPowerSet.m" + str(motor_n), power)

def set_motor_control_mode(crazyflie, mode):
    """Mode must be either AUTOMATIC_CONTROL (0) or MANUAL_CONTROL (1)"""
    if mode != MANUAL_CONTROL and mode != AUTOMATIC_CONTROL:
        raise ValueError("Motor control mode must be either automatic or manual")
    
    crazyflie.cf.param.set_value("motorPowerSet.enable", mode)

    if mode == MANUAL_CONTROL:
        print("Crazyflie motors have been set to manual control mode")
    elif mode == AUTOMATIC_CONTROL:
        print("Crazyflie motors have been set to automatic control mode")
    
def get_motor_control_mode(crazyflie):
    return crazyflie.cf.param.get_value("motorPowerSet.enable")

def stop_motors(crazyflie):
    set_motor_power(crazyflie=crazyflie, motor_n=1, power=0, percent=1)
    set_motor_power(crazyflie=crazyflie, motor_n=2, power=0, percent=1)
    set_motor_power(crazyflie=crazyflie, motor_n=3, power=0, percent=1)
    set_motor_power(crazyflie=crazyflie, motor_n=4, power=0, percent=1)
    print("All motor PWM values are now set to 0")
    

### CONVERSION ###
def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (clockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = - math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians


### TESTS ###

def motor_test(scf):

    stop_motors(scf)
    
    if get_motor_control_mode(scf) == AUTOMATIC_CONTROL:
        print("Motor control mode is currently automatic")
    else:
        print("Motor control mode is currently manual")
    
    set_motor_control_mode(scf, MANUAL_CONTROL)


    try:
        for power in range(0, 101, 5):
            set_motor_power(crazyflie=scf, motor_n=1, power=power, percent=True)
            set_motor_power(crazyflie=scf, motor_n=2, power=power, percent=True)
            set_motor_power(crazyflie=scf, motor_n=3, power=power, percent=True)
            set_motor_power(crazyflie=scf, motor_n=4, power=power, percent=True)
            print("Motor power: {0}%".format(power))
            time.sleep(1)

        stop_motors(scf)
        time.sleep(0.02)
        
        set_motor_control_mode(scf, AUTOMATIC_CONTROL)

    except KeyboardInterrupt("Program interrupted by user"):
        stop_motors(scf)
        set_motor_control_mode(scf, AUTOMATIC_CONTROL)
        

### THREADS ###
def read_user_input():
    global user_input
    while 1:
        if user_input.empty():
            key = readchar.readkey()
            user_input.put(key)

        #"0" button press must exit the program; all threads must quit when pressed
        if key == "0":
            break
 

def main_program():
    cflib.crtp.init_drivers()
    sensors = Sensor_logging()


    with SyncCrazyflie(link_uri=URI, cf=Crazyflie(rw_cache="./cache")) as scf:
        scf.wait_for_params()

        ##### CREATE LOGGING ENVIRONMENT #####
        # # DECLARE & LOAD LOGCONFIG CONTENTS # #
        drone_state_log = LogConfig(name="12states", period_in_ms=10)
        drone_state_log.add_variable(name="stateEstimateZ.x", fetch_as="int16_t")
        drone_state_log.add_variable(name="stateEstimateZ.y", fetch_as="int16_t")
        drone_state_log.add_variable(name="stateEstimateZ.z", fetch_as="int16_t")

        drone_state_log.add_variable(name="stateEstimateZ.vx", fetch_as="int16_t")
        drone_state_log.add_variable(name="stateEstimateZ.vy", fetch_as="int16_t")
        drone_state_log.add_variable(name="stateEstimateZ.vz", fetch_as="int16_t")

        #drone_state_log.add_variable(name="stateEstimateZ.ax", fetch_as="int16_t")
        #drone_state_log.add_variable(name="stateEstimateZ.ay", fetch_as="int16_t")
        #drone_state_log.add_variable(name="stateEstimateZ.az", fetch_as="int16_t")

        drone_state_log.add_variable(name="stateEstimateZ.quat", fetch_as="uint32_t")

        drone_state_log.add_variable(name="stateEstimateZ.rateRoll", fetch_as="int16_t")
        drone_state_log.add_variable(name="stateEstimateZ.ratePitch", fetch_as="int16_t")
        drone_state_log.add_variable(name="stateEstimateZ.rateYaw", fetch_as="int16_t")

        drone_state_log.add_variable(name="pm.batteryLevel", fetch_as="uint8_t")

        scf.cf.log.add_config(drone_state_log)

        # # ADD CALLBACKS # #
        drone_state_log.data_received_cb.add_callback(sensors.battery_log_cb)
        drone_state_log.data_received_cb.add_callback(sensors.position_cb)
        drone_state_log.data_received_cb.add_callback(sensors.velocity_cb)
        #drone_state_log.data_received_cb.add_callback(sensors.acceleration_cb)
        drone_state_log.data_received_cb.add_callback(sensors.attitude_cb)
        drone_state_log.data_received_cb.add_callback(sensors.angular_velocity_cb)

        # # START LOGGING # #
        drone_state_log.start()
        time.sleep(0.1) #wait for the logging to actually start

        sensors.set_origin() #set the origin of the world frame as the starting position of the Crazyflie
        time.sleep(0.1) #wait for the origin measurement to affects position



        ##### PRE-FLIGHT CHECKS #####
        # # MINIMUM BATTERY CHECK # #
        """if (sensors.battery_level <= 20):
            while (1):
                cmd = input("Battery level low ({0}%): do you wish to proceed anyways?\n[Y/N]: ".format(sensors.battery_level))
                if cmd == "Y" or cmd == "y":
                    break
                elif cmd == "N" or cmd == "n":
                    exit()
                else:
                    print ("Invalid input")"""

        # # SENSORS CHECK # #
        if (not sensors.x and not sensors.vx and not sensors.roll):
            print("Sensors measurements are dead!!")
            sensors.print_all()
            exit()

        # # SET MOTOR CONTROL MODE # #
        stop_motors(scf)
    
        if get_motor_control_mode(scf) == AUTOMATIC_CONTROL:
            print("Motor control mode is currently automatic")
        else:
            print("Motor control mode is currently manual")
        
        set_motor_control_mode(scf, MANUAL_CONTROL)



        ##### DECLARE LOOP VARIABLES #####
        # # SET TRIM (HOVER) VARIABLES # #
        thrust_trim = HOVER_THRUST
        roll_trim = 0
        pitch_trim = 0
        yaw_rate_trim = 0

        # # SET TARGET COORDINATES # #
        #this is used when following a pre-existing flight plan
        flight_plan_coordinates = [0, 0, 0] #mm
        flight_plan_velocity = [0, 0, 0] #mm #NOTE: seems to only be used for yaw_CMD determination

        # # SET USER COORDINATES # #
        #this is used to let the user control the drone with the pc keyboard
        user_coordinates = [0, 0, 0] #mm
        user_yaw = [0] #in degrees #NOTE:only used if user control is active

        # # OTHER VARIABLES & FLAGS # #
        motor_PWM = [0, 0, 0, 0]

        pre_PID_time = 0
        outer_PID_time = 0
        inner_PID_time = 0
        update_motors_time = 0
        post_PID_time = 0
        cycle_time = 0

        global user_input
        user_control = False



        ##### PID INITIALIZATION #####
        #NOTE: tau parameter (for the derivative filter) is set to default value
        MAX_ANGLE = math.radians(20) #clamp max angle to avoid divergence from the linerized model
        MAX_RATE = math.radians(20) #clamp max angle rates to avoid sudden movements

        # # EXTERNAL/COMMAND LOOP # #
        along_PID_params = {"Kp":0.2, "Ki":0, "Kd":0, "T":T}
        across_PID_params = {"Kp":0.2, "Ki":0, "Kd":0, "T":T}
        yaw_PID_params = {"Kp":6, "Ki":0, "Kd":0, "T":T, "tolerance":300}
        height_PID_params = {"Kp":2, "Ki":0.5, "Kd":0, "T":T}

        along_PID = PID(**along_PID_params)
        across_PID = PID(**across_PID_params)
        yaw_PID = PID(**yaw_PID_params)
        height_PID = PID(**height_PID_params)
        
        # # INTERNAL/ATTITUDE LOOP # #
        roll_PID_params = {"Kp":6, "Ki": 3, "Kd":0, "T":T, "tolerance":300}
        pitch_PID_params = {"Kp":6, "Ki": 3, "Kd":0, "T":T, "tolerance":300}
        roll_rate_PID_params = {"Kp":25, "Ki": 5, "Kd":2.5, "T":T}
        pitch_rate_PID_params = {"Kp":25, "Ki": 5, "Kd":2.5, "T":T}
        yaw_rate_PID_params = {"Kp":12, "Ki": 16.7, "Kd":0, "T":T}

        roll_PID = PID(**roll_PID_params)
        pitch_PID = PID(**pitch_PID_params)
        roll_rate_PID = PID(**roll_rate_PID_params)
        pitch_rate_PID = PID(**pitch_rate_PID_params)
        yaw_rate_PID = PID(**yaw_rate_PID_params)

        

        ##### CONTROL LOOP #####
        while 1:
            cycle_start = time.process_time()

            time_buoy = time.process_time() #time snapshot

            clear_screen()

            #unreliable since the voltage fluctuates dramatically during flight
            """if sensors.battery_level <= 10:
                print("Battery is dangerously low, shutting motors")
                break"""
            

            ##### MANAGE USER INPUT #####
            if user_input.full():
                key = user_input.get()

                # # EXIT CONDITION # #
                if key == "0":
                    break
                
                # # START/STOP USER CONTROL # #
                if key == readchar.key.ENTER:
                    if user_control == False:
                        user_control = True
                        #when user takes control, preserve the state of the drone to avoid sudden motions/crashes
                        user_coordinates = [target_x, target_y, target_z]
                        user_yaw = yaw_CMD
                    else:
                        user_control = False
                        #TODO what to do about target coordinates when user deactivates manual control?

                # # UPDATE USER COORDINATES # #
                if user_control == True:
                #x axis (positive forward)
                    if key == "w":
                        user_coordinates[0] += 100 #add 10 cm forward
                    elif key == "s":
                        user_coordinates[0] -= 100 #add 10 cm back
                    #y axis (positive left)
                    elif key == "a":
                        user_coordinates[1] += 100 #add 10 cm left
                    elif key == "d":
                        user_coordinates[1] -= 100 #add 10 cm right
                    #z axis (positive up)
                    elif key == readchar.key.SPACE:
                        user_coordinates[2] += 100 #add 10 cm up
                    elif key == "c":
                        user_coordinates[2] -= 100 #add 10 cm down
                    #yaw (positive left)
                    elif key == "q":
                        user_yaw += 9 #add 9 degrees left
                    elif key == "e":
                        user_yaw -= 9 #add 9 degress right



            ##### MANAGE PIDS #####
            # # # COMPUTE TARGET AND DISTANCE FROM IT # # #
            if user_control == False:
                target_x, target_y, target_z = flight_plan_coordinates
                target_vx, target_vy, target_vz = flight_plan_velocity
                yaw_CMD = math.degrees(math.atan2(target_vy, target_vx))

            else:
                target_x, target_y, target_z = user_coordinates
                yaw_CMD = user_yaw

            #distance between drone and target in world coordinates
            delta_x = target_x - sensors.x
            delta_y = target_y - sensors.y
            #same distance expressed along and across the velocity of the target point
            along = delta_x * math.cos(math.radians(yaw_CMD)) + delta_y * math.sin(math.radians(yaw_CMD))
            across = delta_x * math.sin(math.radians(yaw_CMD)) - delta_y * math.cos(math.radians(yaw_CMD))

            #time elapsed before PID computation
            pre_PID_time = time.process_time() - time_buoy
            time_buoy = time.process_time()

            



            # # # UPDATE COMMAND/OUTER PIDS # # #
            #computing the deviation from the hover condition
            roll_CMD = along_PID.update(input=along, target=0) #target is 0 because we want the distance to converge to 0
            pitch_CMD = across_PID.update(input=across, target=0)
            yaw_rate_CMD = yaw_PID.update(input=sensors.yaw, target=yaw_CMD)
            thrust_CMD = height_PID.update(input=sensors.z, target=target_z)

            #adding the hover state
            roll_CMD += roll_trim
            pitch_CMD += pitch_trim
            yaw_rate_CMD += yaw_rate_trim
            thrust_CMD = int(thrust_CMD + thrust_trim)
            
            #time elapsed to convert position into angles & thrust CMD
            outer_PID_time = time.process_time() - time_buoy
            time_buoy = time.process_time()



            # # # UPDATE ATTITUDE/INNER PIDS # # #
            roll_rate_CMD = roll_PID.update(input=sensors.roll, target=roll_CMD)
            pitch_rate_CMD = pitch_PID.update(input=sensors.pitch, target=pitch_CMD)

            #despite being called roll/pitch/yaw rates, they are actually the angular velocities
            thrust_PWM = int(thrust_CMD)
            roll_PWM = int(roll_rate_PID.update(input=sensors.p, target=roll_rate_CMD))
            pitch_PWM = int(pitch_rate_PID.update(input=-sensors.q, target=pitch_rate_CMD))
            yaw_PWM = int(yaw_rate_PID.update(input=-sensors.r, target=-yaw_rate_CMD))          

            #time elapsed to convert angles & thrust commands into PWM commands
            inner_PID_time = time.process_time() - time_buoy
            time_buoy = time.process_time()



            ##### SEND MOTOR PWM COMMANDS #####
            motor_PWM[0] = thrust_PWM - roll_PWM + pitch_PWM  + yaw_PWM
            motor_PWM[1] = thrust_PWM - roll_PWM - pitch_PWM  - yaw_PWM
            motor_PWM[2] = thrust_PWM + roll_PWM - pitch_PWM  + yaw_PWM
            motor_PWM[3] = thrust_PWM + roll_PWM + pitch_PWM  - yaw_PWM

            #clamp PWM to avoid saturation or dangerous behaviour at max thrust
            for i in range (0,4):
                if motor_PWM[i] > MAX_UINT16: motor_PWM[i] = MAX_UINT16
                if motor_PWM[i] < 0: motor_PWM[i] = 0

            #attuate the PWM command
            #set_motor_power(scf, 1, power=motor_PWM[0])
            #set_motor_power(scf, 2, power=motor_PWM[1])
            #set_motor_power(scf, 3, power=motor_PWM[2])
            #set_motor_power(scf, 4, power=motor_PWM[3])

            #time elapsed to send the PWM command to the cf
            update_motors_time = time.process_time() - time_buoy
            time_buoy = time.process_time()

            

            if user_control == True:
                print("WARNING: user control is active, ENTER to deactivate.")
            else:
                print("To activate manual control press ENTER.\nBe sure to know the control scheme before doing so.")
            print("")
                
            sensors.print_all()
            print("")

            print("Target position: ({0:.1f}, {1:.1f}, {2:.1f}) mm; target velocity: ({3:.1f}, {4:.1f}, {5:.1f}) mm/s"
                    .format(target_x, target_y, target_z, target_vx, target_vy, target_vz))

            print("Distance from target\nX: {0:.1f} mm, Y: {1:.1f} mm, Z: {2:.4f} mm ({3:.1f} mm across and {4:.1f} mm along)"
                    .format(delta_x, delta_y, target_z-sensors.z, across, along))
            print("")

            print("Outer loop PIDs")
            print("Thrust: {0} PWM ({1} hover + {2} delta), roll: {3:.4f}°, pitch: {4:.4f}°, yaw: {5:.4f}°"
                    .format(thrust_CMD, HOVER_THRUST, thrust_CMD-HOVER_THRUST, roll_CMD, pitch_CMD, yaw_CMD))
            print("Angle errors -> roll {0:.4f}°, pitch: {1:.4f}°, yaw: {2:.4f}°"
                  .format(roll_PID.error, pitch_PID.error, yaw_PID.error))
            print("")
            
            print("Inner loop PIDs")
            print("Rate commands -> roll rate: {0:.3f}°, pitch rate: {1:.3f}°, yaw rate: {2:.4f}"
                    .format(roll_rate_CMD, pitch_rate_CMD, yaw_rate_CMD))
            print("Angle rate errors -> roll rate: {0:.4f}°/s, pitch rate: {1:.4f}°/s, yaw rate: {2:.4f}°/s"
                    .format(math.degrees(roll_rate_PID.error/1000), math.degrees(pitch_rate_PID.error/1000), math.degrees(yaw_rate_PID.error/1000)))
            print("")

            print("Motor 1 at ", motor_PWM[0], " PWM (thrust: ", thrust_PWM, " roll: ", -roll_PWM, " pitch: ", pitch_PWM, " yaw: ", yaw_PWM, ")")
            print("Motor 2 at ", motor_PWM[1], " PWM (thrust: ", thrust_PWM, " roll: ", -roll_PWM, " pitch: ", -pitch_PWM, " yaw: ", -yaw_PWM, ")")
            print("Motor 3 at ", motor_PWM[2], " PWM (thrust: ", thrust_PWM, " roll: ", roll_PWM, " pitch: ", -pitch_PWM, " yaw: ", yaw_PWM, ")")
            print("Motor 4 at ", motor_PWM[3], " PWM (thrust: ", thrust_PWM, " roll: ", roll_PWM, " pitch: ", pitch_PWM, " yaw: ", -yaw_PWM, ")")
            print("")

            print("Loop time: {0:.6f} s (loop start {1:.6f} + outer PIDs: {2:.6f} + inner PIDs: {3:.6f} + motors: {4:.6f} + loop end: {5:.6f})"
                    .format(cycle_time, pre_PID_time, outer_PID_time, inner_PID_time, update_motors_time, post_PID_time))
            print("")


            #input ("") #DEBUG: remove "#" to look at the cycle step by step
            #break      #DEBUG: remove "#" to check first loop only

            post_PID_time = time.process_time() - time_buoy

            ##### ELAPSE STEP TIME #####
            cycle_time = time.process_time() - cycle_start
            time.sleep(T-cycle_time)


        time.sleep(2*T) #to make sure there is no overlap with the previous thrust command
        stop_motors(scf)
        time.sleep(0.1) #give ample time to the CF to actually stop the motors
        set_motor_control_mode(scf, AUTOMATIC_CONTROL)
        time.sleep(0.1)
        exit()


if __name__ == "__main__":

    read_thread = threading.Thread(target=read_user_input)
    main_thread = threading.Thread(target=main_program)

    main_thread.start()
    read_thread.start()

    main_thread.join()
    read_thread.join()

