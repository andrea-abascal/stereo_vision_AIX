import sys
import time


# Create varaibles for fps calculation
prevTime = 0
newTime = 0

def fps_calculation():
    
    # Display fps
    
    newTime = time.time()
    fps = 1/(newTime - prevTime)
    prevTime = newTime
    fps_text = 'FPS: {:.2f}'.format(fps)
    return fps_text