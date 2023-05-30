import sys
import time




def fps_calculation():
    # Create varaibles for fps calculation
    prevTime = 0
    newTime = 0
    # Display fps
    
    newTime = time.time()
    fps = 1/(newTime - prevTime)
    prevTime = newTime
    fps_text = 'FPS: {:.2f}'.format(fps)
    return fps_text