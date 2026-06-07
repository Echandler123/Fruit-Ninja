# Fruit Ninja CV

A real-time computer-vision hand-tracking game built with OpenCV and MediaPipe: slice falling fruit by moving your index finger in front of your webcam.

## How it works

Each frame is captured from the webcam with OpenCV and passed to MediaPipe's hand-landmarker model, which returns the 21 hand landmarks. The game reads the index-fingertip landmark and maps its normalized coordinates back to pixel coordinates in the frame. A fruit sprite falls down the screen, and collision detection measures the distance between the fingertip and the fruit's center — if it falls within the hit radius, the fruit counts as sliced. The sliced fruit sprite and the score are alpha-blended onto the frame using its RGBA transparency channel, and the result is mirrored and displayed.

## Run it

Dependencies:

```
pip install opencv-python mediapipe
```

The game expects a `data/` directory containing the fruit sprites (`Orange.png`, `Orange_slice_1.png`, `Orange_slice_2.png`) and the MediaPipe hand-landmarker model file (`hand_landmarker.task`).

Run it from the project root:

```
python Game.py
```

A webcam is required. Press `q` to quit.
