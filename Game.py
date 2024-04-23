"""
A game that uses hand tracking to 
hit and destroy green circle enemies.

@author: Nandhini Namasivayam
@version: March 2024

edited from: https://i-know-python.com/computer-vision-game-using-mediapipe-and-python/
"""

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import random
import time
import pygame

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

class fruit:
   def __init__(self, color, screen_width=600, screen_height=400):
       self.color = color
       self.screen_width = screen_width
       self.screen_height = screen_height
       self.respawn()
       self.sprite = cv2.imread('Orange.jpg', -1)
       print("done")
   def respawn(self):
       """
       Selects a random location on the screen to respawn
       """
       self.x = random.randint(50, self.screen_width)
       self.y =  self.screen_height - 50   
    
   
      
class Game:
    def __init__(self):
        # Load game elements
        self.score = 0
        self.level = 0
        #Initialize 100 enemy objects into an array
        # Create the hand detector
        base_options = BaseOptions(model_asset_path='data/hand_landmarker.task')
        options = HandLandmarkerOptions(base_options=base_options,
                                                num_hands=2)
        self.detector = HandLandmarker.create_from_options(options)
        #start timer
        self.start_time = time.time()


        # TODO: Load video
        self.video = cv2.VideoCapture(1)
        frame = self.video.read()[1]
        self.green_enemy = fruit(GREEN)

    
    def draw_landmarks_on_hand(self, image, detection_result):
        """
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

    
    def check_enemy_intercept(self, finger_x, finger_y, enemy, image):
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
        distance = ((finger_x - enemy.x)**2 + (finger_y - enemy.y)**2)**0.5
        #Determines if the finger position overlaps with the enemy's position.
        if distance  < 25:
            #Respawns the enemy
            enemy.respawn()
            #Draws the enemy
            enemy.draw(image)
            #Increases the score
            self.score += 1
            print(self.score)
        pass

    def check_enemy_kill(self, image, detection_result):
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

            #get cordinates of just index finger
            finger = hand_landmarks[HandLandmarkPoints.INDEX_FINGER_TIP.value]

            #map the cordinates back to screen dimensions
            pixelCoord = DrawingUtil._normalized_to_pixel_coordinates(finger.x, finger.y, imageWidth, imageHeight)  

            #get cordinates of just thumb

            #map the cordinates back to screen dimensions
           
            if pixelCoord:
                #draw a green circle around the index finger
                cv2.circle(image,(pixelCoord[0],pixelCoord[1]),25, GREEN, 5)
                #draw a red circle around the thumb 
    
                self.check_enemy_intercept(pixelCoord[0], pixelCoord[1], self.green_enemy, image)


                
    
    def run(self):
        """
        Main game loop. Runs until the 
        user presses "q".
        """    
        # TODO: Modify loop condition  
        while self.video.isOpened():
            # Get the current frame
            frame = self.video.read()[1]

            # Convert it to an RGB image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #the image comes mirrored - flip it
            image = cv2.flip(image, 1)
            self.green_enemy.draw(image)
 
            if self.level == 0 and self.score == 10:
                    #end the game
                    end_time = time.time()
                    print("Time taken to kill 10 enemies: ", end_time - self.start_time)
                    self.video.release
                    self.level = 1
            

            #draw the a enemy object from the array on the image every 3 seconds while keeping the old enemies
            if self.level == 1 and time.time() - self.start_time > 1:
                self.enemies.append(fruit(GREEN))

                i = 0
                while(i< len(self.enemies)):
                    enemy = self.enemies[i]
                    enemy.draw(image)
                    i += 1
                self.start_time = 0
            

            #draw score onto screen
            cv2.putText(image, str(self.score), (50, 50), fontFace= cv2.FONT_HERSHEY_SIMPLEX,fontScale= 1,color = GREEN,thickness = 2)

            # Convert the image to a readable format and find the hands
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(to_detect)

            # Draw the hand landmarks
            #self.draw_landmarks_on_hand(image, results)
            self.check_enemy_kill(image, results)
            # Change the color of the frame back
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("Hand Tracking", image)
            # Break the loop if the user presses 'q'
            if cv2.waitKey(50) & 0xFF == ord('q'):
                print(self.score)
                break
        #Add an option to the game that allows users to play in a timed mode. It would be helpful to add
        #an instance variable like self.level to keep track of what version of the game the user is
        #playing.
        #See how long it takes the user to kill 10 enemies. You can use the time class that’s imported and
        #the time.time() method. End the program when 10 enemies are killed and display the time in the
        #console.
            
        self.video.release
        cv2.destroyAllWindows()
    
        


if __name__ == "__main__":        
    g = Game()
    g.run()