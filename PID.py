import math


"""
  r  +   e   +-----+  u   +-----------+      y
 ---> O ---> | PID | ---> | Attuators | ------->
    - ^      +-----+      +-----------+     |
      | x                                   |
      |    +-----------+     +---------+    |
      '--- | Filtering | <-- | Sensors | <--'
           +-----------+     +---------+
"""


#u(t) = Kp * e(t) + Ki * ∫e(t)*dt + KD * d e(t)/dt * e^(-t/τ)/τ
#e(t) = r(t) - y(t)

#U(s)/E(s) = Kp  + Ki * 1/s + Kd * s/sτ+1

#s = 2/T * (z-1)/(z+1)

#u[n] = p[n] + i[n] + d[n]
#p[n] = Kp * e[n]
#i[n] = Ki*T/2 * (e[n] + e[n-1]) + i[n-1]
#d[n] = 2*Kd/(2τ+T) * (e[n] + e[n-1]) + (2τ-T)/(2τ+T) * d[n-1]


MAX_UINT16 = 65535 #maximum unsigned integer (used for PWM/motor control)


#NOTE: deprecated!
#values are taken from the crazyflie default configuration (cfclient -> tuning)
class DefaultPIDparams:
    def __init__(self, T):
        self.x = {"Kp":5, "Ki":5, "Kd":5, "T":T, "limMin":0., "limMax":MAX_UINT16, "limMinInt":-MAX_UINT16, "limMaxInt":MAX_UINT16}
        self.y = {"Kp":5, "Ki":5, "Kd":5, "T":T, "limMin":0., "limMax":MAX_UINT16, "limMinInt":-MAX_UINT16, "limMaxInt":MAX_UINT16}
        self.z = {"Kp":5, "Ki":5, "Kd":5, "T":T, "limMin":0., "limMax":MAX_UINT16, "limMinInt":-MAX_UINT16, "limMaxInt":MAX_UINT16}
        self.roll = {"Kp":5, "Ki":5, "Kd":5, "T":T, "limMin":0., "limMax":MAX_UINT16, "limMinInt":-MAX_UINT16, "limMaxInt":MAX_UINT16}
        self.pitch = {"Kp":5, "Ki":5, "Kd":5, "T":T, "limMin":0., "limMax":MAX_UINT16, "limMinInt":-MAX_UINT16, "limMaxInt":MAX_UINT16}
        self.yaw = {"Kp":5, "Ki":5, "Kd":5, "T":T, "limMin":0., "limMax":MAX_UINT16, "limMinInt":-MAX_UINT16, "limMaxInt":MAX_UINT16}
        self.vx = {"Kp":25, "Ki":25, "Kd":5, "T":T, "limMin":0., "limMax":MAX_UINT16, "limMinInt":-MAX_UINT16, "limMaxInt":MAX_UINT16}
        self.vy = {"Kp":25, "Ki":25, "Kd":5, "T":T, "limMin":0., "limMax":MAX_UINT16, "limMinInt":-MAX_UINT16, "limMaxInt":MAX_UINT16}
        self.vz = {"Kp":25, "Ki":25, "Kd":5, "T":T, "limMin":0., "limMax":MAX_UINT16, "limMinInt":-MAX_UINT16, "limMaxInt":MAX_UINT16}
        self.rollRate = {"Kp":500, "Ki":500, "Kd":5, "T":T, "limMin":0., "limMax":MAX_UINT16, "limMinInt":-MAX_UINT16, "limMaxInt":MAX_UINT16}
        self.pitchRate = {"Kp":500, "Ki":500, "Kd":5, "T":T, "limMin":0., "limMax":MAX_UINT16, "limMinInt":-MAX_UINT16, "limMaxInt":MAX_UINT16}
        self.yawRate = {"Kp":100, "Ki":50, "Kd":5, "T":T, "limMin":0., "limMax":MAX_UINT16, "limMinInt":-MAX_UINT16, "limMaxInt":MAX_UINT16}



class PID:
    def __init__(self, Kp=10, Ki=2, Kd=1, T=1, tau=0.001,
                 limMin = 0, limMax = math.inf,
                 limMinInt = -math.inf, limMaxInt = math.inf, tolerance=0):

        #control coefficients
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        #step time
        self.T = T #should be less than System_BW/10

        #low-pass derivative filter to reject HF noise
        self.tau = tau 

        #output clamping
        self.limMin = limMin 
        self.limMax = limMax

        #integral clamping
        self.limMinInt = limMinInt
        self.limMaxInt = limMaxInt

        #integrator
        self.integrator = 0
        self.prevError = 0 #for integrator

        #differentiator; derivative of Input instead of Error to avoid jumps when the Target changes
        self.differentiator = 0
        self.prevInput = 0 #for differentiator

        #for angle re-wrapping
        self.tolerance = tolerance
        self.prevTarget = 0
        self.flip_counter = 0
        self.target_flip_counter = 0
        self.input_flip_counter = 0

        #saved just for external ease of access
        self.error = 0
        self.output = 0
    
    def update (self, input, target):
        #If input or target represent angles, +/- 180° is a critical angle,
        #since, if increased, the angle will "flip" with a 360° excursion.
        #To avoid this behaviour we must compensate with a counter-rotation, times the number of flips.

        #if tolerance is set to 0 skip the above considerations
        if self.tolerance != 0: 
            #count number of flips, if any
            if math.abs(target-self.prevTarget) > self.tolerance:
                if target > self.prevTarget:
                    self.target_flip_counter += 1
                else:
                    self.target_flip_counter -= 1

            if math.abs(input-self.prevInput) > self.tolerance:
                if input > self.prevInput:
                    self.input_flip_counter += 1
                else:
                    self.input_flip_counter -= 1
            
            #compensate the flips
            target = target - self.target_flip_counter * 360
            input = input - self.input_flip_counter * 360



        #error
        self.error = target - input

        #proportional
        proportional = self.Kp * self.error

        #integral
        self.integrator = self.integrator + self.Ki * self.T * (self.error + self.prevError)/2

        #anti-wind-up using integrator clamping
        if (self.integrator > self.limMaxInt):
            self.integrator = self.limMaxInt
        elif (self.integrator < self.limMinInt):
            self.integrator = self.limMinInt

        #differentiator
        self.differentiator = - (2 * self.Kd * (input - self.prevInput) + (2 * self.tau - self.T) * self.differentiator) / (2 * self.tau + self.T)

        #output
        self.output = proportional + self.integrator + self.differentiator

        if (self.output > self.limMax):
            self.output = self.limMax
        elif (self.output < self.limMin):
            self.output = self.limMin

        #store error, input & target for PID.update() call
        self.prevError = self.error
        self.prevInput = input
        self.prevTarget = target

        return self.output