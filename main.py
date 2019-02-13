### Imported Libraries ###

import time                             # Sleep/interrupt librariy
import cv2                              # OpenCV library
import numpy as np                      # Python Math Library
import os                               # Operating system library
from PIL import Image                   # Image manipulation and conversion library
import RPi.GPIO as GPIO                 # Raspberry PI GPIO library (Pinouts)
from threading import Timer, Thread     # Python threading library
import screeninfo

### PROGRAM STATES ###
START_TIME = time.time()

CURENT_STATE = 0                        # Current State
DETECT_STATE = 0                        # Face Detection State
PROCES_STATE = 1                        # Image Processing State
RECOGN_STATE = 2                        # Face Recognition State

BUZZER_STATE = 1

### PROGRAM CONSTANTS ###

THREAD_LIST = []
VID_CAP_THREAD_NUM = -1
IMG_TRN_THREAD_NUM = -1


IS_PROCESSING_FINISHED = False          # Boolean: True if image processing state is finished or not
COUNT_FACES_REGISTERED = 0              # Integer: Number of registered faces

FONT = cv2.FONT_HERSHEY_SIMPLEX
### GPIO CONSTANTS ###
HIGH = 1                                # Integer: GPIO signal is HIGH/ON
LOW = 0                                 # Integer: GPIO signal is LOW/OFF

GPIO_BUTTON_1 = 26                      # Integer: GPIO BCM 26 is used for the button
GPIO_BUZZER = 23                        # Integer: GPIO BCM 23 is used for the buzzer

### GPIO SETUP ###
GPIO.setmode(GPIO.BCM)                  # Initalizes the GPIO python driver to use BCM GPIOs
GPIO.setup(GPIO_BUTTON_1, GPIO.IN)      # Initializes the use of the button
GPIO.setup(GPIO_BUZZER, GPIO.OUT)       # Initializes the use of the buzzer

def buzz():
    '''
        Activates the buzzer through the GPIO
    '''
    count = 0
    while(count < 3):
        GPIO.output(GPIO_BUZZER,GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(GPIO_BUZZER,GPIO.LOW)
        count += 1

def GPIO_button_1_callback(channel):
    '''
        This function is activated when an event in GPIO_BUTTON_1 is triggered, and is used to reset the program from
        face recognition state to face detection state.

        Args: 
            channel (int): The channel number based on the numbering system you have specified - BOARD or BCM).
        
        Returns:
            None
    '''
    global start
    global end

    global LOW
    global HIGH
    global CURENT_STATE
    global DETECT_STATE
    global COUNT_FACES_REGISTERED
    global IS_PROCESSING_FINISHED
    global BUZZER_STATE

    max_elapsed_time = 3
    tap_buzz_time = 1


    if GPIO.input(GPIO_BUTTON_1) == HIGH:
       start = time.time()
    if GPIO.input(GPIO_BUTTON_1) == LOW:
       end = time.time()
       elapsed = end - start
       if elapsed <= tap_buzz_time:
           if(BUZZER_STATE == 1):
               BUZZER_STATE = 0
           else:
               BUZZER_STATE = 1
       if(elapsed >= max_elapsed_time):
           COUNT_FACES_REGISTERED = 0
           IS_PROCESSING_FINISHED = False
           CURENT_STATE = DETECT_STATE

### GPIO EVENT LISTENER register_faces ###    
GPIO.add_event_detect(GPIO_BUTTON_1, GPIO.BOTH, callback = GPIO_button_1_callback, bouncetime=100)


### PROGRAM THREADS ###

class VideoCaptureThread(Thread):
    '''
        This class is responsible for continuously reading the camera feed without interruption from other processes.

        Args:
            Thread (Object): Python thread object to implement parallel processing.

    '''
    def __init__(self, src):
        '''
            Initializes the class.
            
            Args: 
                src (int): Video Capture Mode
        '''
        Thread.__init__(self)
        self.cap = cv2.VideoCapture(src)
        self.cap.release()
        self.cap = cv2.VideoCapture(src)
        
        screen = screeninfo.get_monitors()[0]
        width, height = screen.width, screen.height
        #self.cap.set(3,640)
        #self.cap.set(4,480)
        
 
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        
        self.window_name = 'frame'
        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
##        cv2.moveWindow(self.window_name, screen.x - 1, screen.y - 1)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.stop = False

    def run(self):
        '''
            Runs the thread for parallel processing.
        '''
        while(self.stop == False):
            self.ret, self.frame = self.cap.read()

    def Stop(self):
        '''
            Stops the run loop and terminates the video capture thread.
        '''
        self.cap.release()
        cv2.destroyAllWindows()
        self.stop = True

    def read(self):
        '''
            Returns the frame read by the camera.

            Returns: self.frame (numpy.ndarray): A matrix representation of the video feed being read.
        '''
    
        return self.ret, self.frame, self.window_name
    
    def getMinWH(self):
        '''
            Returns the minimum width and height of the video feed.

            Returns: minW (int): Integer scale factor for width
                     minH (int): Integer scale factor for height
        '''

        minW = 0.1 * self.cap.get(3)
        minH = 0.1 * self.cap.get(4)
        return minW, minH
    
class ImageProcessingThread (Thread):
    '''
        This class is responsible for preocessing trained images in parallel.
    '''
    def __init__(self, face_cascade, recognizer, path):
        '''
            Initializes the thread

            Args: 
                face_cascade (CascadeClassifier): Contains the trained Haar Cascades.
                recognizer (cv2.FaceRecognizer): Contains the algorithm for recognizing images
                path (str): directory path of where the images are stored
        '''
        Thread.__init__(self)
        self.face_cascade = face_cascade
        self.recognizer = recognizer
        self.path = path
        
    def run(self):
        '''
            Runs the thread to train images
        '''
        train_images(self.face_cascade,self.recognizer, self.path)



def getImagesAndLabels(path, face_cascade):
    '''
        Retrieves the images including its labels in preparation for training

        Args:
            path (str): directory path of where the images are stored
            face_cascade (CascadeClassifier): Contains the trained Haar Cascades.

        Returns:
            faceSamples(list<numpy.Array>): Array representation of an image
            ids(list<int>): Array of ids corresponding to the user

    '''
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = face_cascade.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

def register_faces(face_cascade, face_id, max_count):
    '''
        Registers user's face in preperation for training

        Args:
            face_id(int): user id input
            max_count(int): maximum number of times a face is to be recognized.
            

        Returns:
            if image counter is greater than or equal to the max number of images to be registered per user
                then -> function returns the next state of the program (image training)
            if not:
                then -> function returns the current state (no change)
    '''

    global THREAD_LIST
    global VID_CAP_THREAD_NUM
    global COUNT_FACES_REGISTERED
    ret, img, window_name = THREAD_LIST[0].read()
    img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
             
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        COUNT_FACES_REGISTERED += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("/home/pi/sauron/dataset/User." + str(face_id) + '.' + str(COUNT_FACES_REGISTERED) + ".jpg", gray[y:y+h,x:x+w])
        img_flip = cv2.imread("/home/pi/sauron/dataset/User." + str(face_id) + '.' + str(COUNT_FACES_REGISTERED) + ".jpg", 0)
        img_flip = cv2.flip(img_flip, 1)
        cv2.imwrite("/home/pi/sauron/dataset/User." + str(face_id) + '.' + str(COUNT_FACES_REGISTERED + max_count) + ".jpg", img_flip)
        cv2.putText(img, str(COUNT_FACES_REGISTERED) + "/" + str(max_count) , (x+5,y-5), FONT, 1, (255,255,255), 2)
        cv2.imshow(window_name, img) 
    if  COUNT_FACES_REGISTERED >= max_count: # Take 50 face sample and stop video
        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        return PROCES_STATE
    
    return CURENT_STATE

def train_images(face_cascade, recognizer, path):
    '''
        Trains the user's image using the Local Binary Patterns Histograms.

        Args:
            face_cascade (CascadeClassifier): Contains the trained Haar Cascades.
            recognizer (cv2.FaceRecognizer): Contains the algorithm for recognizing images
            path (str): Directory where the images are saved
    '''
    # Path for face image database
    
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path, face_cascade)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))



def detect_faces(recognizer,face_cascade):
    '''
        Recongizes the user's faces in real time

        Args:
            face_cascade (CascadeClassifier): Contains the trained Haar Cascades.
            recognizer (cv2.FaceRecognizer): Contains the algorithm for recognizing images
    '''

    # Define min window size to be recognized as a face
    global START_TIME
    global THREAD_LIST
    global VID_CAP_THREAD_NUM
    global BUZZER_STATE

    names = ['Registered User'] 
    id = 0
    ret, img, window_name =THREAD_LIST[VID_CAP_THREAD_NUM].read()
    minW, minH = THREAD_LIST[VID_CAP_THREAD_NUM].getMinWH()
    img = cv2.flip(img, -1) # Flip vertically

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale( 
        gray,
        scaleFactor = 1.3,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    
        
    for(x,y,w,h) in faces:
        
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        threshold = 100
        # Check if confidence is less than 100 ==> "0" is perfect match
        if (confidence <= threshold):
            START_TIME = time.time()
            id = names[id]
            #confidence = "  {0}%".format(round(100 - confidence))
            confidence_str = "  {0}%".format(round((threshold - confidence)/ threshold * 100))
        else:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
            id = "Unknown User"
            #confidence = "  {0}%".format(round(100 - confidence))
            confidence_str = "  {0}%".format(round((threshold - confidence)/ threshold * 100))
        
        print(confidence)
        cv2.putText(img, str(id), (x+5,y-5), FONT, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence_str), (x+5,y+h-5), FONT, 1, (255,255,0), 1)  
    
        cv2.imshow('frame',img)
    if len(faces) == 0 or confidence > threshold:

        if ((time.time() - START_TIME) > 3.0):
            if(BUZZER_STATE == 1):
                thread_buzz = Thread(target=buzz)
                thread_buzz.start()
                START_TIME = time.time()
        #threading.Timer(1.0, buzz()).start()

def main():
    #Initalize GPIO.output as 
    GPIO.output(GPIO_BUZZER,GPIO.LOW)

    global THREAD_LIST
    global VID_CAP_THREAD_NUM
    global IMG_TRN_THREAD_NUM
    global CURENT_STATE
    global IS_PROCESSING_FINISHED
    face_cascade = cv2.CascadeClassifier('/home/pi/sauron/haarcascade_frontalface_default.xml') #Haar Cascade used for Face Detection
    recognizer = cv2.face.LBPHFaceRecognizer_create()                           #Local Binarry Patterns Histograms

    path = '/home/pi/sauron/dataset/'

    vid_capture_thread = VideoCaptureThread(0)
    vid_capture_thread.start()
    time.sleep(2)
    THREAD_LIST.append(vid_capture_thread)
    VID_CAP_THREAD_NUM = len(THREAD_LIST) - 1
    while(True):
        ret, img, window_name = THREAD_LIST[VID_CAP_THREAD_NUM].read()
        
        if CURENT_STATE == DETECT_STATE:
            face_id = 0
            max_count = 80
            CURENT_STATE = register_faces(face_cascade, face_id, max_count)
        elif CURENT_STATE == PROCES_STATE:
            if(not IS_PROCESSING_FINISHED):
                img_process_thread = ImageProcessingThread(face_cascade, recognizer, path)
                img_process_thread.start()
                time.sleep(0.5)
                THREAD_LIST.append(img_process_thread)
                IMG_TRN_THREAD_NUM = len(THREAD_LIST) - 1
                IS_PROCESSING_FINISHED = True
                
            if not THREAD_LIST[IMG_TRN_THREAD_NUM].isAlive():
                CURENT_STATE = RECOGN_STATE
        elif CURENT_STATE == RECOGN_STATE:
            START_TIME = time.time()
            detect_faces(recognizer,face_cascade)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            THREAD_LIST[VID_CAP_THREAD_NUM].Stop()
            GPIO.cleanup()
            break
        img = cv2.flip(img, -1)
        cv2.imshow(window_name, img)

if __name__ == "__main__":
    main()