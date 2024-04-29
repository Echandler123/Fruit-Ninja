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

    
    def check_fruit_intercept(self, finger_x, finger_y,fruitx,fruity):
        """
        Determines if the finger position overlaps with the 
        enemy's position. Respawns and draws the enemy and 
        increases the score accordingly.
        Args:
            finger_x (float): x-coordinates of index finger
            finger_y (float): y-coordinates of index finger
            image (_type_): The image to draw on
        """
        print("f")
        print(finger_x)
        print("f")
        print(finger_y)
        # Calculate the distance between the finger and the enemy
        distance = ((finger_x - fruitx)**2 + (finger_y - fruity)**2)**0.5
        print("distance")
        print(distance)
        #Determines if the finger position overlaps with the enemy's position.
        if distance < 200:
            #Draws the enemy
            self.score += 1
            print(self.score)
            hit = True
        pass

    def check_fruit_kill(self, image, detection_result,fruitx,fruity):
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


            #map the cordinates back to screen dimensions
            if pixelCoord:
                #draw a green circle around the index finger
                cv2.circle(image,(pixelCoord[0],pixelCoord[1]),25, GREEN, 5)
                #draw a red circle around the thumb 
                self.check_fruit_intercept(pixelCoord[0], pixelCoord[1],fruitx,fruity)


                
    
    def run(self,hit):
        """
        Main game loop. Runs until the 
        user presses "q".
        """    
        # TODO: Modify loop condition
        x= random.randint(0, 640)
        y = 0  
        while self.video.isOpened():
            Orange = cv2.imread('data/Orange.png', -1)
            # Get the current frame
            frame = self.video.read()[1]
            
            if self.score < 1:
                frame = self.video.read()[1]
        # Where to place the cowboy hat on the screen
                y1, y2 =  y , y + Orange.shape[0]
                x1, x2 = x, x + Orange.shape[1]
                y = y + 1
                
    

        # Saving the alpha values (transparencies)
                alpha = Orange[:, :, 3] / 255.0

        # Overlays the image onto the frame (Don't change this)
                for c in range(0, 3):
                    frame[y1:y2, x1:x2, c] = (alpha * Orange[:, :, c] +
                                            (1.0 - alpha) * frame[y1:y2, x1:x2, c])
        
        # Display the resulting frame
            cv2.imshow('Orange', frame)
            

            # Convert it to an RGB image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #the image comes mirrored - flip it
            if self.level == 0 and self.score == 10:
                    #end the game
                end_time = time.time()
                print("Time taken to kill 10 enemies: ", end_time - self.start_time)
                self.video.release
                self.level = 1
            

            #draw score onto screen
        

            # Convert the image to a readable format and find the hands
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(to_detect)
            fy =  y + y + Orange.shape[0]
            fx = x + x + Orange.shape[1]
            fy = fy/2
            fx = fx/2
            print(fx)
            print(fy)
            self.check_fruit_kill(image, results,fx,fy)
            image = cv2.flip(image, 1)
            cv2.putText(image, str(self.score), (50, 50), fontFace= cv2.FONT_HERSHEY_SIMPLEX,fontScale= 1,color = GREEN,thickness = 2)

            # Draw the hand landmarks
            #self.draw_landmarks_on_hand(image, results)
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
    g.run(False)