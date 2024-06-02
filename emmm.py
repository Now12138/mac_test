from r_leg_test.r_leg_test import PCA9685
import time
import math
import smbus

pwm1 = PCA9685(0x40, debug=True)
pwm1.setPWMFreq(50)
pwm2 = PCA9685(0x42, debug=True)
pwm2.setPWMFreq(50)
global REST
REST=[[100,100,5,150,65,115],[150,100,15,105,35,165]]

def trans(angle):
    return (angle*100/9 + 500)

def rest_position(rest):
    global rear_shoulder_left_rest_angle
    global rear_leg_left_rest_angle
    global rear_feet_left_rest_angle
    global rear_shoulder_right_rest_angle
    global rear_leg_right_rest_angle
    global rear_feet_right_rest_angle
    global front_shoulder_left_rest_angle
    global front_leg_left_rest_angle
    global front_feet_left_rest_angle
    global front_shoulder_right_rest_angle
    global front_leg_right_rest_angle
    global front_feet_right_rest_angle
    rear_shoulder_left_rest_angle = REST[0][0]
    rear_leg_left_rest_angle = REST[0][1]
    rear_feet_left_rest_angle = REST[0][2]
    rear_shoulder_right_rest_angle = REST[0][3]
    rear_leg_right_rest_angle = REST[0][4]
    rear_feet_right_rest_angle = REST[0][5]
    front_shoulder_left_rest_angle = REST[1][0]
    front_leg_left_rest_angle = REST[1][1]
    front_feet_left_rest_angle = REST[1][2]
    front_shoulder_right_rest_angle = REST[1][3]
    front_leg_right_rest_angle = REST[1][4]
    front_feet_right_rest_angle = REST[1][5]

def body_move_body_up_and_down(raw_value):
    global rear_shoulder_left_rest_angle
    global rear_leg_left_rest_angle
    global rear_feet_left_rest_angle
    global rear_shoulder_right_rest_angle
    global rear_leg_right_rest_angle
    global rear_feet_right_rest_angle
    global front_shoulder_left_rest_angle
    global front_leg_left_rest_angle
    global front_feet_left_rest_angle
    global front_shoulder_right_rest_angle
    global front_leg_right_rest_angle
    global front_feet_right_rest_angle
    range = 10
    range2 = 15
    
    if raw_value > 0:
        rear_leg_left_rest_angle -= range
        rear_feet_left_rest_angle += range2
        rear_leg_right_rest_angle += range
        rear_feet_right_rest_angle -= range2
        front_leg_left_rest_angle -= range
        front_feet_left_rest_angle += range2
        front_leg_right_rest_angle += range
        front_feet_right_rest_angle -= range2

    elif raw_value < 0:
        rear_leg_left_rest_angle += range
        rear_feet_left_rest_angle -= range2
        rear_leg_right_rest_angle -= range
        rear_feet_right_rest_angle += range2
        front_leg_left_rest_angle += range
        front_feet_left_rest_angle -= range2
        front_leg_right_rest_angle -= range
        front_feet_right_rest_angle += range2
            
def body_move_position_1():
    global rear_shoulder_left_rest_angle
    global rear_leg_left_rest_angle
    global rear_feet_left_rest_angle
    global rear_shoulder_right_rest_angle
    global rear_leg_right_rest_angle
    global rear_feet_right_rest_angle
    global front_shoulder_left_rest_angle
    global front_leg_left_rest_angle
    global front_feet_left_rest_angle
    global front_shoulder_right_rest_angle
    global front_leg_right_rest_angle
    global front_feet_right_rest_angle
    variation_leg = 15
    variation_feet1 = 20
    variation_feet2 = 10
        
    #self.servo_front_shoulder_left.angle = self.servo_front_shoulder_left_rest_angle - 10 + move
    front_leg_left_rest_angle -= variation_leg
    front_feet_left_rest_angle -= variation_feet1

def body_move_position_2():
    global rear_shoulder_left_rest_angle
    global rear_leg_left_rest_angle
    global rear_feet_left_rest_angle
    global rear_shoulder_right_rest_angle
    global rear_leg_right_rest_angle
    global rear_feet_right_rest_angle
    global front_shoulder_left_rest_angle
    global front_leg_left_rest_angle
    global front_feet_left_rest_angle
    global front_shoulder_right_rest_angle
    global front_leg_right_rest_angle
    global front_feet_right_rest_angle
    variation_leg = 15
    variation_feet1 = 20
    variation_feet2 = 10
        
    front_feet_left_rest_angle += variation_feet1

    front_feet_right_rest_angle -= variation_feet2

    rear_feet_left_rest_angle += variation_feet2

    rear_feet_right_rest_angle -= variation_feet2

def body_move_position_3():
    global rear_shoulder_left_rest_angle
    global rear_leg_left_rest_angle
    global rear_feet_left_rest_angle
    global rear_shoulder_right_rest_angle
    global rear_leg_right_rest_angle
    global rear_feet_right_rest_angle
    global front_shoulder_left_rest_angle
    global front_leg_left_rest_angle
    global front_feet_left_rest_angle
    global front_shoulder_right_rest_angle
    global front_leg_right_rest_angle
    global front_feet_right_rest_angle
    variation_leg = 15
    variation_feet1 = 20
    variation_feet2 = 10
        
    #front_shoulder_left.angle = self.servo_front_shoulder_left_rest_angle - 10 + move
    front_leg_right_rest_angle += variation_leg
    front_feet_right_rest_angle += variation_feet1+variation_feet2

def body_move_position_4():
    global rear_shoulder_left_rest_angle
    global rear_leg_left_rest_angle
    global rear_feet_left_rest_angle
    global rear_shoulder_right_rest_angle
    global rear_leg_right_rest_angle
    global rear_feet_right_rest_angle
    global front_shoulder_left_rest_angle
    global front_leg_left_rest_angle
    global front_feet_left_rest_angle
    global front_shoulder_right_rest_angle
    global front_leg_right_rest_angle
    global front_feet_right_rest_angle
    variation_leg = 15
    variation_feet1 = 20
    variation_feet2 = 10
        
    front_feet_right_rest_angle -= variation_feet1
        
    rear_feet_left_rest_angle += variation_feet2

    rear_feet_right_rest_angle -= variation_feet2

def body_adjust():
    global rear_shoulder_left_rest_angle
    global rear_leg_left_rest_angle
    global rear_feet_left_rest_angle
    global rear_shoulder_right_rest_angle
    global rear_leg_right_rest_angle
    global rear_feet_right_rest_angle
    global front_shoulder_left_rest_angle
    global front_leg_left_rest_angle
    global front_feet_left_rest_angle
    global front_shoulder_right_rest_angle
    global front_leg_right_rest_angle
    global front_feet_right_rest_angle
    variation_leg = 15
    front_leg_right_rest_angle -= variation_leg
    front_leg_left_rest_angle += variation_leg
        
def body_move_position_5():
    global rear_shoulder_left_rest_angle
    global rear_leg_left_rest_angle
    global rear_feet_left_rest_angle
    global rear_shoulder_right_rest_angle
    global rear_leg_right_rest_angle
    global rear_feet_right_rest_angle
    global front_shoulder_left_rest_angle
    global front_leg_left_rest_angle
    global front_feet_left_rest_angle
    global front_shoulder_right_rest_angle
    global front_leg_right_rest_angle
    global front_feet_right_rest_angle
    variation_leg = 15
    variation_feet1 = 20
    variation_feet2 = 10

    #rear_leg_left_rest_angle -= variation_leg
    rear_feet_left_rest_angle -= variation_feet2*2
        
def body_move_position_6():
    global rear_shoulder_left_rest_angle
    global rear_leg_left_rest_angle
    global rear_feet_left_rest_angle
    global rear_shoulder_right_rest_angle
    global rear_leg_right_rest_angle
    global rear_feet_right_rest_angle
    global front_shoulder_left_rest_angle
    global front_leg_left_rest_angle
    global front_feet_left_rest_angle
    global front_shoulder_right_rest_angle
    global front_leg_right_rest_angle
    global front_feet_right_rest_angle
    variation_leg = 15
    variation_feet1 = 20
    variation_feet2 = 10
        
    #rear_leg_left_rest_angle += variation_leg
        
    #rear_leg_right_rest_angle += variation_leg
    rear_feet_right_rest_angle += variation_feet2*2

def move():
    global rear_shoulder_left_rest_angle
    global rear_leg_left_rest_angle
    global rear_feet_left_rest_angle
    global rear_shoulder_right_rest_angle
    global rear_leg_right_rest_angle
    global rear_feet_right_rest_angle
    global front_shoulder_left_rest_angle
    global front_leg_left_rest_angle
    global front_feet_left_rest_angle
    global front_shoulder_right_rest_angle
    global front_leg_right_rest_angle
    global front_feet_right_rest_angle
    pwm2.setServoPulse(0,trans(front_shoulder_left_rest_angle))
    pwm2.setServoPulse(1,trans(front_leg_left_rest_angle))
    pwm2.setServoPulse(2,trans(front_feet_left_rest_angle))
    pwm2.setServoPulse(3,trans(front_shoulder_right_rest_angle))
    pwm2.setServoPulse(4,trans(front_leg_right_rest_angle))
    pwm2.setServoPulse(5,trans(front_feet_right_rest_angle))
        
    pwm1.setServoPulse(0,trans(rear_shoulder_left_rest_angle))
    pwm1.setServoPulse(1,trans(rear_leg_left_rest_angle))
    pwm1.setServoPulse(2,trans(rear_feet_left_rest_angle))
    pwm1.setServoPulse(3,trans(rear_shoulder_right_rest_angle))
    pwm1.setServoPulse(4,trans(rear_leg_right_rest_angle))
    pwm1.setServoPulse(5,trans(rear_feet_right_rest_angle))

def walk():
    body_move_position_1()
    move()
    time.sleep(0.2)
                    
    body_move_position_2()
    move()
    time.sleep(0.2)
                    
    body_move_position_3()
    move()
    time.sleep(0.2)
                    
    body_move_position_4()
    move()
    time.sleep(0.2)
                    
    body_adjust()
    move()
    time.sleep(0.2)
                    
    body_move_position_5()
    move()
    time.sleep(0.2)
                    
    body_move_position_6()
    move()


rear_shoulder_left_rest_angle = REST[0][0]
rear_leg_left_rest_angle = REST[0][1]
rear_feet_left_rest_angle = REST[0][2]
rear_shoulder_right_rest_angle = REST[0][3]
rear_leg_right_rest_angle = REST[0][4]
rear_feet_right_rest_angle = REST[0][5]
front_shoulder_left_rest_angle = REST[1][0]
front_leg_left_rest_angle = REST[1][1]
front_feet_left_rest_angle = REST[1][2]
front_shoulder_right_rest_angle = REST[1][3]
front_leg_right_rest_angle = REST[1][4]
front_feet_right_rest_angle = REST[1][5]
move()
x="start"
while (True):
    if(x=="start"):
        print("----------Command input prompt:-----------")
        print("Enter r: enter a resting state")
        print("Enter w: Walk forward")
        print("Input u: Center of gravity raised")
        print("Input l: Center of gravity decrease")
        print("Enter q: exit the program")
        rest_position(REST)
    elif (x=="w"):
        walk()
    elif (x=="r"):
        rest_position(REST)
        move()
    elif (x=="l"):
        body_move_body_up_and_down(-1)
        move()
    elif (x=="u"):
        body_move_body_up_and_down(1)
        move()
    elif (x=="q"):
        break
    else:
        print("This command is invalid, please re-enter:",end='')
        pass
    x = input()



#time.sleep(2)
#rest_position(REST)
print(trans(135))
#time.sleep(2)
#standing_position(REST)
rest_position(REST)
    
