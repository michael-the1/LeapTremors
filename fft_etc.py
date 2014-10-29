import os, sys, inspect, thread, time

sys.path.insert(0,r'c:\Program Files (x86)\Leap Motion\LeapSDK\samples\lib\x86') 
sys.path.insert(0,r'c:\Program Files (x86)\Leap Motion\LeapSDK\samples\lib') 
import Leap
import numpy as np
import matplotlib.pyplot as plt

from Leap import CircleGesture, KeyTapGesture, ScreenTapGesture, SwipeGesture
from numpy import fft

x = []
y = []
y2 = []
count = 0

class SampleListener(Leap.Listener):
    
    
    def on_connect(self, controller):
        print "Connected"


    def on_frame(self, controller):
        frame = controller.frame()
        hand = frame.hands[0]
        
 #       print "Frame id: %d  timestamp: %d  hand position: %f  finger position: %f"% (
 #               frame.id, frame.timestamp, hand.palm_position.y, hand.fingers[1].joint_position(3).y)
        
        print "Frame id: %d  timestamp: %d  relative finger position: %f"% (
                frame.id, frame.timestamp, hand.palm_position.y - hand.fingers[1].joint_position(3).y)        
        global x,y,y2,count
        count= count+1
        x.append(frame.timestamp)
        y.append(hand.palm_position.y - hand.fingers[1].joint_position(3).y)
        y2.append(hand.palm_position.y)
        
def main():
    # Create a sample listener and controller
    listener = SampleListener()
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)

    # Keep this process running until Enter is pressed
    print "Press Enter to quit..."
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        controller.remove_listener(listener)
        print("x", x)
        print("y", y2)
        
        x[:]= [t - x[0] for t in x]
        
        plt.figure(1)
        plt.subplot(211)
        plt.plot(x,y2)
        
        # plt.subplot(212)
        # plt.plot(x,y)
        
        # plt.figure(2)
        # plt.subplot(311)
        # xf = fft.fftfreq(len(y), x[len(x)-1]-x[0])
        # mask = (xf > 0) & (xf <= 10) # show this frequency range 
        # xf = xf[mask]

        
        plt.subplot(212)
        yf= fft.fft(y)
        
        xf = fft.fftfreq(len(yf))
        print(xf.min(), xf.max())
        
        # find the peak in the coefficients
        ayf = np.abs(yf)**2
        idx= np.argmax(ayf[1:]) # to make sure 0 isn't the outcome of idx(because xf[0] is always 0 TODO: beter uitleggen dit.
        print("ayf", ayf)
        print("max value of ayf %f" % ayf.max())
        print("first value of ayf %f" % ayf[0])
        print("lenght of xf %d" % len(xf))
        print("idx: %d" % idx)
        freq= xf[idx]
        frate = count/(x[len(x)-1]/1000000.0)
        print("totale tijd: %f" % (x[len(x)-1]/1000000.0))
        print("framerate: %f" % frate)
        print("frequency: %f" % freq)
        hertz=abs(freq*frate)
        print("hertz: %f" % hertz)
        
        plt.plot(x,np.abs(yf))
        
        plt.grid()
        plt.show()        
        
if __name__ == "__main__":
    main()
