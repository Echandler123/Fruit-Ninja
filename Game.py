# Elijah Chandler
# 5/18/24
import cv2
import random
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# Score text color. Green is the middle channel, so it is identical in BGR and
# RGB; the unused RED/BLUE constants (which were RGB, not the BGR OpenCV expects)
# have been removed.
SCORE_COLOR = (0, 255, 0)

# --- Gameplay tuning -------------------------------------------------------
FALL_INCREMENT = 20      # pixels the fruit drops each frame
HIT_RADIUS = 100         # max fingertip-to-fruit distance (px) that counts as a slice
FRAME_WIDTH = 1280       # assumed frame width used for the spawn math
BOTTOM_Y = 480           # y at which an unsliced fruit resets to the top
SPAWN_EDGE_GAP = 100     # horizontal padding kept clear of the frame edges

# Original hardcoded bounds for the very first spawn (kept as-is so the game
# plays identically; later respawns use the image-derived bounds below).
FIRST_SPAWN_MIN = 407
FIRST_SPAWN_MAX = FRAME_WIDTH - 411  # 869

# When a fruit is sliced it is drawn as two halves: the main sprite shifts left
# and a smaller piece flies to the right.
SLICE_LEFT_OFFSET = 50
SLICE_RIGHT_OFFSET = 100

# Library Constants
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkPoints = mp.solutions.hands.HandLandmark
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
DrawingUtil = mp.solutions.drawing_utils


class Fruit:
    """A single falling fruit: its position, motion, and sliced state.

    The whole-fruit sprite drives the collision/spawn math; the smaller piece
    sprite only affects how far in from the edges a respawn can land.
    """

    def __init__(self, whole_sprite, piece_sprite):
        self.height, self.width = whole_sprite.shape[:2]
        piece_h, piece_w = piece_sprite.shape[:2]
        # Respawn bounds keep the whole fruit and its flying piece on screen.
        self.spawn_min = piece_h + SPAWN_EDGE_GAP + self.width
        self.spawn_max = FRAME_WIDTH - (piece_w + SPAWN_EDGE_GAP + self.width)
        # First spawn uses the original fixed bounds.
        self.x = random.randint(FIRST_SPAWN_MIN, FIRST_SPAWN_MAX)
        self.y = 0
        self.sliced = False

    def fall(self):
        """Drop the fruit one step."""
        self.y += FALL_INCREMENT

    def respawn(self):
        """Send the fruit back to the top at a new random x."""
        self.x = random.randint(self.spawn_min, self.spawn_max)
        self.y = 0

    def center(self):
        """Center point of the whole-fruit sprite, used for hit detection."""
        return (self.x + self.width / 2, self.y + self.height / 2)


class Game:
    def __init__(self):
        # Load game elements
        self.score = 0
        # Load the fruit sprites once (RGBA) instead of re-reading from disk
        # every frame inside the game loop.
        self.fruit_whole = cv2.imread('data/Orange.png', -1)
        self.fruit_piece = cv2.imread('data/Orange_slice_1.png', -1)
        self.fruit_sliced = cv2.imread('data/Orange_slice_2.png', -1)
        # Create the hand detector
        base_options = BaseOptions(model_asset_path='data/hand_landmarker.task')
        options = HandLandmarkerOptions(base_options=base_options,
                                                num_hands=2)
        self.detector = HandLandmarker.create_from_options(options)
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

    def check_fruit_intercept(self, finger_x, finger_y, fruit_x, fruit_y):
        """
        Determines whether the fingertip is close enough to the fruit's
        center to count as a slice.
        Args:
            finger_x (float): x-coordinate of the index fingertip
            finger_y (float): y-coordinate of the index fingertip
            fruit_x (float): x-coordinate of the fruit's center
            fruit_y (float): y-coordinate of the fruit's center
        """
        # Distance between the fingertip and the fruit's center
        distance = ((finger_x - fruit_x) ** 2 + (finger_y - fruit_y) ** 2) ** 0.5
        return distance < HIT_RADIUS

    def check_fruit_kill(self, image, detection_result, fruit_x, fruit_y):
        """
        Maps the detected index fingertip back to pixel coordinates and checks
        whether it intercepts the fruit.
        Args:
            image (Image): The image the hand was detected in
            detection_result (HandLandmarkerResult): HandLandmarker detection results
            fruit_x (float): x-coordinate of the fruit's center
            fruit_y (float): y-coordinate of the fruit's center
        """
        image_height, image_width = image.shape[:2]
        hand_landmarks_list = detection_result.hand_landmarks
        # Check the first detected hand's index fingertip.
        for hand_landmarks in hand_landmarks_list:
            finger = hand_landmarks[HandLandmarkPoints.INDEX_FINGER_TIP.value]
            # Map the normalized landmark back to screen pixels.
            pixel_coord = DrawingUtil._normalized_to_pixel_coordinates(
                finger.x, finger.y, image_width, image_height)
            if pixel_coord:
                return self.check_fruit_intercept(
                    pixel_coord[0], pixel_coord[1], fruit_x, fruit_y)
            return False
        return False

    def run(self):
        """
        Main game loop. Runs until the user presses "q".
        """
        fruit = Fruit(self.fruit_whole, self.fruit_piece)
        while self.video.isOpened():
            # Get the current frame
            frame = self.video.read()[1]

            # Capture where the fruit is drawn this frame before it moves: the
            # sprite is rendered at the position from the end of the previous
            # frame, then the fruit advances.
            draw_x, draw_y = fruit.x, fruit.y
            fruit.fall()

            # Score and respawn: a sliced fruit scores a point, and either a
            # sliced fruit or one that fell off the bottom returns to the top.
            if fruit.sliced:
                self.score += 1
                print(self.score)
                fruit.respawn()
            elif fruit.y == BOTTOM_Y:
                fruit.respawn()
                print(self.score)

            # Render the fruit onto the frame. A sliced fruit is drawn as two
            # halves; otherwise the whole fruit is drawn.
            if fruit.sliced:
                self.overlay(frame, self.fruit_sliced,
                             draw_y, draw_x - SLICE_LEFT_OFFSET)
                if draw_x + fruit.width - SLICE_LEFT_OFFSET >= SLICE_RIGHT_OFFSET:
                    self.overlay(frame, self.fruit_piece,
                                 draw_y, draw_x + SLICE_RIGHT_OFFSET)
            else:
                self.overlay(frame, self.fruit_whole, draw_y, draw_x)

            fruit.sliced = False

            # Convert to RGB and detect the hand(s).
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(to_detect)

            # Test the fingertip against the fruit's center for the next frame.
            center_x, center_y = fruit.center()
            fruit.sliced = self.check_fruit_kill(image, results, center_x, center_y)

            # Mirror the view, draw the score, and display.
            image = cv2.flip(image, 1)
            cv2.putText(image, str(self.score), (50, 50),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=SCORE_COLOR, thickness=2)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("Fruit Ninja", image)

            # Break the loop if the user presses 'q'
            if cv2.waitKey(50) & 0xFF == ord('q'):
                print("Final Score:")
                print(self.score)
                break
        self.video.release()
        cv2.destroyAllWindows()

    def overlay(self, frame, sprite, top, left):
        """Alpha-blend an RGBA sprite onto the frame with its top-left at
        (left, top)."""
        height, width = sprite.shape[:2]
        bottom, right = top + height, left + width
        alpha = sprite[:, :, 3] / 255.0
        for c in range(0, 3):
            frame[top:bottom, left:right, c] = (
                alpha * sprite[:, :, c] +
                (1.0 - alpha) * frame[top:bottom, left:right, c])
        return frame


if __name__ == "__main__":
    g = Game()
    g.run()
