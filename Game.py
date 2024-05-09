# Elijah Chandler
# 5/18/24
import cv2
import random
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import time
import math

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Library Constants
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkPoints = mp.solutions.hands.HandLandmark
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
DrawingUtil = mp.solutions.drawing_utils
    
class Game:
    def __init__(self):
        # Load game elements
        self.score = 0
        self.level = 0
        # Create the hand detector
        base_options = BaseOptions(model_asset_path='data/hand_landmarker.task')
        options = HandLandmarkerOptions(base_options=base_options,
                                                num_hands=2)
        self.detector = HandLandmarker.create_from_options(options)
        #start timer
        self.start_time = time.time()
        # Loads video
        self.video = cv2.VideoCapture(1)
    
    def draw_landmarks_on_hand(self, image, detection_result):
        """
        Code from the FingerTrackingGame lab
        Draws all the landmarks on the hand
        Args:
            image (Image): Image to draw on
            detection_result (HandLandmarkerResult): HandLandmarker detection results
        """
        # Get a list of the landmarks
        hand_landmarks_list = detection_result.hand_landmarks
        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            # Save the landmarks into a NormalizedLandmarkList
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            # Draw the landmarks on the hand
            DrawingUtil.draw_landmarks(image,
                                       hand_landmarks_proto,
                                       solutions.hands.HAND_CONNECTIONS,
                                       solutions.drawing_styles.get_default_hand_landmarks_style(),
                                       solutions.drawing_styles.get_default_hand_connections_style())
    
    def check_fruit_intercept(self, finger_x, finger_y, fruitx, fruity):
        """
        Determines if the finger position overlaps with the 
        enemy's position. Respawns and draws the enemy and 
        increases the score accordingly.
        Args:
            finger_x (float): x-coordinates of index finger
            finger_y (float): y-coordinates of index finger
            image (_type_): The image to draw on
        """
        # Calculate the distance between the finger and the enemy
        distance = ((finger_x - fruitx)**2 + (finger_y - fruity)**2)**0.5
        self.hit = False
        # Determines if the finger position overlaps with the enemy's position and returns true if so
        if distance < 100:
            self.hit = True
        return self.hit

    def check_fruit_kill(self, image, detection_result, fruitx, fruity):
        """
        Draws a green circle on the index finger 
        and calls a method to check if we've intercepted
        with the enemy
        Args:
            image (Image): The image to draw on
            detection_result (HandLandmarkerResult): HandLandmarker detection results
        """
        # Get image details
        imageHeight, imageWidth = image.shape[:2]
        hand_landmarks_list = detection_result.hand_landmarks
        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx] 
            # Get cordinates of just index finger
            finger = hand_landmarks[HandLandmarkPoints.INDEX_FINGER_TIP.value]
            # Map the cordinates back to screen dimensions
            pixelCoord = DrawingUtil._normalized_to_pixel_coordinates(finger.x, finger.y, imageWidth, imageHeight)  
            self.hit = False
            # Return if the finger hit the fruit
            if pixelCoord:
                self.hit = self.check_fruit_intercept(pixelCoord[0], pixelCoord[1],fruitx,fruity)
            return self.hit           
    
    def run(self):
        """
        Main game loop. Runs until the 
        user presses "q".
        """    
        x_boundary  = 407
        y_boundary = 1280 - 411
        # Generates where the fruit will first spawn
        x = random.randint(x_boundary, y_boundary)
        y = 0  
        self.hit = False
        inc = 20
        while self.video.isOpened():
            # Load the fruit images
            orange = cv2.imread('data/Orange.png', -1)
            orange2 = cv2.imread('data/Orange_slice_1.png', -1)
            # Get the current frame
            frame = self.video.read()[1]
            # Calculating the courdinates of the fruit
            y1, y2 =  y , y + orange.shape[0]
            x1, x2 = x, x + orange.shape[1]
            y3,y4 = y, y + orange2.shape[0]
            y = y + inc
            x3,x4 = x, x + orange2.shape[1]
            # When the fruit is hit increases the score and calculates the courdinates for the split fruit images 
            # and the spawn courdinates for a new fruit
            if self.hit == True:
                y = 0
                self.score = self.score + 1
                print(self.score)
                orange = cv2.imread('data/Orange_slice_2.png', -1)
                x2 -= 50
                x1 -= 50
                x3 = x3 + 100
                x4 = x4 + 100
                x = random.randint((orange2.shape[0] + 100 + orange.shape[1]), 1280 - (orange2.shape[1] + 100 + orange.shape[1]))
            # When the fruit reaches the bottom of the screen it creates the new spawn courdinates for the fruit
            elif y == 480:
                x = random.randint((orange2.shape[0] + 100 + orange.shape[1]), 1280 -  (orange2.shape[1] + 100 + orange.shape[1]))
                y = 0
                self.hit = False 
                print(self.score)  
            # Saving the alpha values (transparencies)
            alpha = orange[:, :, 3] / 255.0
            self.overlay(y1, y2, x1, x2, alpha, orange, frame)
            # Display the resulting frame
            cv2.imshow('Orange', frame)
            # Display the resulting frame for the second image of the split fruit if the fruit was hit
            if self.hit == True and x2 >= 100:
                alpha = orange2[:, :, 3] / 255.0
                self.overlay(y3, y4, x3, x4, alpha, orange2, frame)
            cv2.imshow('Orange2', frame)
            self.hit = False 
            # Convert it to an RGB image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the image to a readable format and find the hands
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(to_detect)
            fy =  y + y + orange.shape[0]
            fx = x + x + orange.shape[1]
            # Calculating middle of the fruit image
            fy = fy/2
            fx = fx/2
            self.hit = self.check_fruit_kill(image, results,fx,fy)
            image = cv2.flip(image, 1)
            cv2.putText(image, str(self.score), (50, 50), fontFace= cv2.FONT_HERSHEY_SIMPLEX,fontScale= 1,color = GREEN,thickness = 2)
            # Change the color of the frame back
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("Fruit Ninja", image)
            # Break the loop if the user presses 'q'
            if cv2.waitKey(50) & 0xFF == ord('q'):
                print("Final Score:")
                print(self.score)
                break   
        self.video.release
        cv2.destroyAllWindows()
    # Overlays the of the fruit onto the frame 
    def overlay(self, y1, y2, x1, x2, alpha,orange, frame):
        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (alpha * orange[:, :, c] +
                                    (1.0 - alpha) * frame[y1:y2, x1:x2, c])
        return frame

if __name__ == "__main__":        
    g = Game()
    g.run()