import cv2
import random
def fruit_overlay():
    # Load the overlay image with an alpha channel (transparency)
    Orange = cv2.imread('data/Orange.png', -1)

    # Capture video from the webcam
    video = cv2.VideoCapture(1)
    x= random.randint(0, 640)
    while True:
        frame = video.read()[1]
        # Where to place the cowboy hat on the screen
        y1, y2 =  50 , 50 + Orange.shape[0]
        x1, x2 = x, x + Orange.shape[1]

        # Saving the alpha values (transparencies)
        alpha = Orange[:, :, 3] / 255.0

        # Overlays the image onto the frame (Don't change this)
        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (alpha * Orange[:, :, c] +
                                    (1.0 - alpha) * frame[y1:y2, x1:x2, c])
        
        # Display the resulting frame
        cv2.imshow('Orange', frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    # Release the capture
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    fruit_overlay()