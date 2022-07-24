mport io
import picamera
import logging
import socketserver
import numpy as np
from threading import Condition
from http import server
import cv2

import socket
import time
import threading

import re
import traceback

import Adafruit_PCA9685
from adafruit_servokit import ServoKit
import math as mat

import cv2
import numpy as np
from imutils.video import VideoStream



log_on=True
#Logging
def log(string):
    if(log_on):
        print(string)


light_pwm = Adafruit_PCA9685.PCA9685(address=0x41, busnum=1)
light_pwm.set_pwm_freq(1000)
kit = ServoKit(channels=16)


camera_resolution = [255,200] #804, 630

point = [0,0]
spotlight = [0,0]
mainlight = [0,0]
colorlight = [0,0]
colors = 0
move_camera = 0
detect_image = [False,False,False,True,False]#


actReq = [0,0,0,0,0,0,0] #move spotlight, spotlight, main light, color, color summary, move camera, detect
order_point = [0,0,0] # X, Y, order
actual_point = [0,0] # X, Y

Xmnoznik = 2#*(200 / 1267)#2
Ymnoznik = 2#*(200 / 659)#3
z=200#100
offsetX = 5#4
offsetY = -30

##### VISUAL #####

PAGE="""\
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <style type="text/css">
    body { margin:0; }
    </style>
</head>
<body>
    <div style="width: 100px; height: 100px; background: red;">
        <img src="stream.mjpg" style="height: 100vh; width: 100vw;">
    </div>
</body>
"""




class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = None
                (h, w) = (None, None)
                zeros = None
                i = 0
                face_cache=()
                tracker = cv2.TrackerMOSSE_create()
                track_frame = None
                no_tracking_success = 0
                print_point = (0,0)
                point_color = (0, 0, 255)
                face_cascade = cv2.CascadeClassifier('/home/pi/Lightheart/haarcascade_frontalface_default.xml')
                print("start")
                while True:
                    with output.condition:
                         output.condition.wait()
                         frame = output.frame
                         if detect_image[3]:
                            face_cache=()
                            i = 0
                            track_frame = None
                            tracker = None
                            tracker = cv2.TrackerMOSSE_create()
                            point_color = (0, 0, 255)
                            cv2.CV_LOAD_IMAGE_COLOR = 1
                            detect_image[4] = False
                            detect_image[3] = False
                            print("clean")

                         if detect_image[0]:
                           
                            frame = decode(frame)
                            print_point = (int(point[0]*camera_resolution[0]/804), int(point[1]*camera_resolution[1]/630))

                            ## search for faces
                            if not i % 2 and detect_image[1]:
                                 face_cache = haar_cascade(frame, face_cascade)
                                 i=i+1

                            ## iterate over face_chache, if point is within face square, initiate tracker and set flag for chasing image to true
                            if len(face_cache) and detect_image[1]:
                                track_frame = iterate_over_faces(face_cache,print_point)
                                for (x,y,w,h) in face_cache:
                                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
                                if track_frame and detect_image[4]:
                                    tracker.init(frame, track_frame)
                                    detect_image[2] = True
                                    detect_image[4] = False
                                i=i+1

 
                            ##  tracking
                            if track_frame is not None and detect_image[2] :
                                (success, box) = tracker.update(frame)
                                if success:
                                    (x, y, w, h) = [int(v) for v in box]
                                    cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
                                    print_point = (int(x + w/2), int(y + h/2))
                                    point_color = (0, 255, 0)
                                    detect_image[1] = False

                                    no_tracking_success = 0
                                    i=0
                                else:
                                    no_tracking_success+=1
                                    print("target lost:  " + str(no_tracking_success))


                                    face_cache = haar_cascade(frame, face_cascade)
                                    if len(face_cache):
                                        for (x,y,w,h) in face_cache:
                                            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)

                                        track_frame = find_closest_face(face_cache,print_point)
                                        tracker = None
                                        tracker = cv2.TrackerMOSSE_create()
                                        tracker.init(frame, track_frame)

                                    if no_tracking_success == 20:
                                        print("-----------target lost ------ " )
                                        no_tracking_success = 0
                                        point_color = (0, 0, 255)
                                        detect_image[0] = False
                                        detect_image[1] = False
                                        detect_image[2] = False
                                        detect_image[3] = True
#TODO włączyć szukanie obrazu i wybrać obiekt nie większyt niż.. w przypadku gdy zostanie zgubiony obiekt... korekcja co pare przesunieć?
#wyciagniecie punktu poza obwod- konczenie sledzenia, ale wyszukiwanie zostaje chwile.
# obraz nie moze byc wyszukany, jeli punkt nie zostanie celowo umieszczony w ramce, dodac flage w handlowaniu move i jesli dotkniecie poza kwadratem- kasowac

                            frame = cv2.circle(frame, print_point, 4, point_color, 2)

                            if i == 20:
                               detect_image[0] = False
                               detect_image[1] = False
                               detect_image[2] = False
                               detect_image[3] = True
                               print("koniec")
                            ##encode
                            _, frame = cv2.imencode('.JPEG', frame)
                            frame = frame.tobytes()

                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()




class VW:
    def __init__(self):
       
        with picamera.PiCamera(resolution='255x200', framerate=9) as camera:

            camera.start_recording(output, format='mjpeg')


    #print(clientsocket)
    #print(address)

    #speed_control_thread = speed_control_thread(2,"Servo")
            print("control")
            try:
                address = ('', 8001)
                server = StreamingServer(address, StreamingHandler)
                app_started = 1
                server.serve_forever()
            finally:
                camera.stop_recording()


output = StreamingOutput()

##### SOCKET CONTROL #####




class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

class control_thread(threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.Port = 8002
        self.maxConnections = 999
        self.sock = socket.socket()
        self.sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('', self.Port))
        self.sock.listen(self.maxConnections)

        self.clientsocket = None
        self.address = None
        (self.clientsocket, self.address) = self.sock.accept()
    def run(self):

        self.receive()
######
# receive encoded message from client and decode them.
# parse sets flag in actReq array indicating the need for specific action and set of values point/spotlight/mainlight/colorlight for said operation
######
    def receive(self):
        global move_to_position, chaser_light, mainlight, color_lights, colors, move_camera, detect_image, speed_controll_thread, app_started

        while True:
            try:
                message = self.clientsocket.recv(50).decode()
                print(message)
                if not message == "":
                    empty_msg = 0
               
                    log("parse result:  " + str(self.parse(message, point, spotlight, colorlight)))
                    if actReq[0] == True:
                    #point[0] = (point[0] * Xmnoznik  - 100) * 1
                    #point[1] = (point[1] * Ymnoznik  - 100) * -1
                    #log("move_to_position result:  " + str(speed_controll_thread.move_to_position(point[0], point[1])))
                   # speed_controll_thread.move_to_position(50, 50)
                        order_point = [point[0], point[1], 1]
                        actReq[0] = False
                        detect_image[4] = True #indicate that it's a new click

                    if actReq[1] == True:
                        log("chaser_light result:  " + str(chaser_light(spotlight[0], spotlight[1])))
                        actReq[1] = False
   
                    if actReq[2] == True:
                        log("main_lights result:  " + str(main_lights(mainlight[0], mainlight[1])))
                        actReq[2] = False

                    if actReq[3] == 1:
                        log("color_lights result:  " + str(color_lights(colorlight[0], colorlight[1])))
                        actReq[3] = False
   
                    if actReq[4] == True:
                        log("color_lights_summary result: " + str(color_lights_summary(colors)))
                        actReq[4] = False

                    if actReq[5] == True:
                        log("camera_move result: " + str(camera_move(move_camera)))
                        actReq[5] = False

                    if actReq[6] == True:
                        log("detect result: " + str(detect(detect)))
                        actReq[6] = False

                else:
                    empty_msg+=1
                    if empty_msg>10 :
                        detect_image[0] = False
                        detect_image[3] = True
                        (self.clientsocket, self.address) = self.sock.accept()
                        print("START AGAIN")
                        #print(clientsocket)
                        #print(address)    
                   
            except:
                print(traceback.format_exc())
                log("Error in receive")
                pass


    def parse (self,string,point,spotlight,colorlight):
        try:
            if(string[0]=='X'):
                print("parse ruch")
                point[0] = re.search(r'X:([0-9]*)', string).group(0)
                point[1] = re.search(r'Y:([0-9]*)', string).group(0)
                if point[0] and point[1]:
                    point[0] = int(point[0][2:])
                    point[1] = int(point[1][2:])
                    speed_control_thread.move_to_position(point[0],point[1])
                    actReq[0] = True
                    return True

            if(string[0]=='L'):
                print("parse spot")
                spotlight[0] = re.search(r'L:([0-9]*)', string).group(0)
                spotlight[1] = re.search(r'S:([0-9]*)', string).group(0)
                if spotlight[0] and spotlight[1]:
                    spotlight[0] = int(spotlight[0][2:])
                    spotlight[1] = int(spotlight[1][2:])
                    actReq[1] = True
                    return True

            if(string[0]=='M'):
                print("parse main")
                mainlight[0] = re.search(r'M:([0-9]*)', string).group(0)
                mainlight[1] = re.search(r'P:([0-9]*)', string).group(0)
                if mainlight[0] and mainlight[1]:
                    mainlight[0] = int(mainlight[0][2:])
                    mainlight[1] = int(mainlight[1][2:])
                    actReq[2] = True
                    return True

            if(string[0]=='C'):
                print("parse color")
                colorlight[0] = re.search(r'C:([0-9]*)', string).group(0)
                colorlight[1] = re.search(r'I:([0-9]*)', string).group(0)
                if colorlight[0] and colorlight[1]:
                    colorlight[0] = int(colorlight[0][2:])
                    colorlight[1] = int(colorlight[1][2:])
                    actReq[3] = True
                    return True

            if(string[0]=='K'):
                print("parse colors summary")
                colors = re.search(r'K:([0-9]*)', string).group(0)
                actReq[4] = True
                return True

            if(string[0]=='H'):
                print("parse move camera")
                move_camera = re.search(r'H:([0-9]*)', string).group(0)
                print(move_camera)
                actReq[5] = True
                return True

            if(string[0]=='D'):
                print("parse detect")
                detect = re.search(r'D:([0-9]*)', string).group(0)
                actReq[6] = True
                return True
            log("partse: signal not recognised")
        except:
            log(traceback.format_exc())
            log("Parse exeption: Error in parse")
            pass



##### SERVO CONTROL #####

Xmnoznik = 2#*(200 / 1267)#2
Ymnoznik = 2#*(200 / 659)#3
z=200#100
offsetX = 0#4
offsetY = -30
class speed_control_thread(threading.Thread):

    old_deg = (90,90)

    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name

        kit.servo[0].angle = self.old_deg[0]
        kit.servo[1].angle = self.old_deg[1]
        #self.order_loop()
        print("speed_control_thread init")

    def move_delay(i):
        t1 = time.time()
        while True:
            t2 = time.time()
            if i <= 0.0001:
                i = 0.0001
                break
            elif t2-t1 > i:
                break
        return i

    def move_to_deg(deg,servo):
        print("------move: "+ str(deg))
        if servo == 0:
            kit.servo[0].angle = deg
        if servo == 1:
            kit.servo[1].angle = deg

    def run(order_point, old_deg):
        print("start run")
        distance = (abs(order_point[0]-old_deg[0]),abs(order_point[1]-old_deg[1]))
        print("distance: " +str(distance) + "  order_point: " + str(order_point) + "  old_deg: " + str(old_deg))

        larger_dist = 0 if distance[0] > distance[1] else 1
        n = distance[larger_dist]
        #print("distance: " +str(distance) + "  dist: " + str(n) + "  old_deg: " + str(old_deg))
        deg = [old_deg[0], old_deg[1]] #Begins from old degrees
        t = 0.1
 ######Both axis
        if distance[0] and distance[1]:# both axis
            print("dwa----")

            for x in range(1,n+1):

                speed_control_thread.move_to_deg(deg[larger_dist],    larger_dist)
                speed_control_thread.move_to_deg(1/(distance[larger_dist]/distance[not larger_dist]) * deg[not larger_dist],    not larger_dist)
                t = speed_control_thread.move_delay(t)

                print("larger:  "+ str(x) + "  not larger:  " + str( 1/(distance[larger_dist]/distance[not larger_dist]) * x) + "  int:  " + str(np.round( 1/(distance[larger_dist]/distance[not larger_dist]) * x))  )  

                if order_point[larger_dist] > old_deg[larger_dist]:
                    print("lewo")
                    if x <= int(n/20):
                        t *= (0.70)
                    elif x >= int(n-(n/5)):
                        t *= (1.210)
                    if deg[larger_dist] >= order_point[larger_dist]:
                        break
                    else:
                        deg[larger_dist]+=1

                elif order_point[larger_dist] < old_deg[larger_dist]:
                    print("prawo")
                    if x <= int(n/20):
                        t *= (0.70)
                    elif x >= int(n-(n/5)):
                        t *= (1.210)
                    if deg[larger_dist] <= order_point[larger_dist]:
                        break
                    else:
                        deg[larger_dist]-=1
### lesser degree
                if order_point[not larger_dist] > old_deg[not larger_dist]:
                    print("lewo")
                    if x <= int(n/20):
                        t *= (0.70)
                    elif x >= int(n-(n/5)):
                        t *= (1.210)
                    if deg[not larger_dist] >= order_point[larger_dist]:
                        break
                    else:
                        deg[not larger_dist]+=1

                elif order_point[not larger_dist] < old_deg[not larger_dist]:
                    print("prawo")
                    if x <= int(n/20):
                        t *= (0.70)
                    elif x >= int(n-(n/5)):
                        t *= (1.210)
                    if deg[not larger_dist] <= order_point[larger_dist]:
                        break
                    else:
                        deg[not larger_dist]-=1


 ######Single axis
        elif bool( distance[larger_dist]) != bool( distance[not larger_dist]):# single axis
            print("jedno---")

            if order_point[larger_dist] > old_deg[larger_dist]:
                print("lewo")
                for x in range(n):

                    speed_control_thread.move_to_deg(deg[larger_dist],larger_dist)
                    t = speed_control_thread.move_delay(t)

                    if x <= int(n/20):
                        print("przyspiesza deg: "+ str(deg[larger_dist]) + "  t:" + str(t))
                        t *= (0.70) #+ 0.5*t #*0.70
                    elif x >= int(n-(n/5)):
                        print("zwalnia deg: "+ str(deg[larger_dist]) + "  t:" + str(t))
                        t*=(1.210)
               
                    else:
                        print("jedzie deg: "+ str(deg[larger_dist]) + "  t:" + str(t))
                    if deg[larger_dist] >= order_point[larger_dist]:
                        break
                    else:
                        deg[larger_dist]+=1

            elif order_point[larger_dist] < old_deg[larger_dist]:
                print("prawo")
                for x in range(n):

                    speed_control_thread.move_to_deg(deg,larger_dist)
                    t = speed_control_thread.move_delay(t)

                    if x <= int(n/20):
                        print("przyspiesza deg: "+ str(deg[larger_dist]) + "  t:" + str(t))
                        t *= (0.70) #+ 0.5*t #*0.70
                    elif x >= int(n-(n/5)):
                        print("zwalnia deg: "+ str(deg[larger_dist]) + "  t:" + str(t))
                        t*=(1.210)
                    else:
                        print("jedzie deg: "+ str(deg[larger_dist]) + "  t:" + str(t))
                    if deg[larger_dist] <= order_point[larger_dist]:
                        break
                    else:
                        deg[larger_dist]-=1


        return True



    def calculate_degree( X1, Y1):
        if X1 > 100 or Y1 > 100 or X1 < -100 or Y1 < -100:
            #log("invalid coordinates:  " + str(X)+","+str(Y))
            return False
      #  log("ODBIOR  "+str(X1)+ "   "+str(Y1))
##X
        print("calculate_degree enter: " + str(X1) + "  Y1: " + str(Y1))
        if X1 < 0:
            A = mat.degrees(mat.atan(abs(X1) / z))
            alpha = 90 + offsetX - A
            print("calculate_degree  X1 < 0:  " + str(alpha))
        if not X1:
            alpha = 90 + offsetX
        if X1 > 0:
            A = mat.degrees(mat.atan(X1 / z))
            alpha = 90 + offsetX + A
            print("calculate_degree  X1 > 0:  " + str(alpha))
##Y
        if Y1 < 0:
            B = mat.degrees(mat.atan(abs(Y1) / z))
            beta = 90 + offsetY - B
        if not Y1:
            beta = 90 + offsetY
        if Y1 > 0:
            B =  mat.degrees(mat.atan(abs(Y1) / z))
            beta = 90 + offsetY + B
#Check
        alpha = int(alpha)
        beta = int(beta)

        if alpha > 180 or alpha < 0 or beta > 180 or beta < 0:
            #log("invalid degree value a="+str(alpha)+" b="+str(beta))
            return False
        print("calculate_degree:  " + str(alpha))
        return (alpha, beta)

    def move_to_position(X1, Y1):
        print("move to position")
        point  = speed_control_thread.calculate_degree(X1, Y1)
        print("move to pos deg: "+str(point)+"  move to pos x,y: "+str(X1)+"  "+str(Y1)+ " old point :"+ str(speed_control_thread.old_deg))
        if point:
            speed_control_thread.run(point, speed_control_thread.old_deg)
            speed_control_thread.old_deg = point
        return True


##### END SERVO CONTROL #####
def chaser_light(L,S):
    if L > 1 or L < 0 or S > 100 or S < 0:
        log("invalid spotlight parameters")
        return False
    if(L):
        light_pwm.set_pwm(0, 0, int(S*40.9))
        return True
    else:
        light_pwm.set_pwm(0, 0, 0)
        return True
    return False

def main_lights(M,P):
    if M > 1 or M < 0 or P > 100 or P < 0:
        log("invalid spotlight parameters")
        return False
    if(M):
        light_pwm.set_pwm(1, 0, int(P*40.9))
        return True
    else:

        light_pwm.set_pwm(1, 0, 0)
        return True
    return False

def color_lights(C,I):
    if C > 4 or C < 1 or I > 100 or I < 0:
        log("invalid spotlight parameters")
        return False
    if(C == 1):
        light_pwm.set_pwm(4, 0, int(I*40.9))
        return True
    elif (C == 2):
        light_pwm.set_pwm(5, 0, int(I*40.9))
        return True
    elif (C == 3):
        light_pwm.set_pwm(6, 0, int(I*40.9))
        return True
    elif (C == 4):
        light_pwm.set_pwm(7, 0, int(I*40.9))
        return True
    return False

def color_lights_summary( a):
    print("colors")
    return True
def camera_move( a):
    print("move")
    return True

def detect( a):
    detect_image[0] = True   #start detecting
    detect_image[1] = True   #detection controlled by tracking
    detect_image[2] = False  #tracking detected image
    detect_image[3] = True   #clean all data
    detect_image[4] = False  #is a new click
    return True

def haar_cascade(frame, face_cascade):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces):
        face_cache = faces
    else:
        face_cache = ()
    print("haar_cascade: " + str(face_cache))
    return face_cache

def iterate_over_faces(face_cache,print_point):
    for face_cache_iterator in face_cache:
        if print_point[0] > face_cache_iterator[0] and print_point[0] < (face_cache_iterator[0]+face_cache_iterator[2])\
        and print_point[1] > face_cache_iterator[1] and print_point[1] < (face_cache_iterator[1]+face_cache_iterator[3]):
            track_frame = (face_cache_iterator[0], face_cache_iterator[1], face_cache_iterator[2], face_cache_iterator[3])
            print("OK")
            return track_frame
        else:
            return None

def find_closest_face(face_cache,print_point):#TODO this needs to be checked if is ok, most likely isn't but for now works
    dist = [None] * len(face_cache)
    n = 0
    for face_cache_iterator in face_cache:

        dist[n] = np.sqrt(np.power(int((face_cache_iterator[0] + face_cache_iterator[2]/2)- print_point[0]),2) + np.power(int((face_cache_iterator[1]+ face_cache_iterator[3]/2)-print_point[1]),2) )                
        print("dist " + str(dist[n]))
        n+=1
    dist_min= dist.index(max(dist))
    track_frame = (face_cache[dist_min][0], face_cache[dist_min][1], face_cache[dist_min][2], face_cache[dist_min][3])
    return track_frame
           
   
    return None

def decode(frame):
    frame = np.frombuffer(frame, dtype=np.int8)
    frame = cv2.imdecode(frame,cv2.CV_LOAD_IMAGE_COLOR)
    return frame
