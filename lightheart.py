##############################################
#         LIGHT HEART         #
##############################################

from LightHeart_lib import *



if __name__ == '__main__':
    control_thread = control_thread(1,"control")
    control_thread.start()
    speed_control_thread = speed_control_thread(2,"Servo")
    #speed_control_thread.start()
    w = VW()


