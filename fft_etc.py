import os, sys, inspect, thread, time

sys.path.insert(0,r'c:\Program Files (x86)\Leap Motion\LeapSDK\samples\lib\x86') 
sys.path.insert(0,r'c:\Program Files (x86)\Leap Motion\LeapSDK\samples\lib') 
import Leap
import numpy as np
import matplotlib.pyplot as plt

# fft from scipy
import scipy
import scipy.fftpack
import pylab
import math

from Leap import CircleGesture, KeyTapGesture, ScreenTapGesture, SwipeGesture
from numpy import fft
from scipy.interpolate import interp1d

x = []
y = []
# y2 = []

class SampleListener(Leap.Listener):
    
    
    def on_connect(self, controller):
        print "Connected"


    def on_frame(self, controller):
        frame = controller.frame()
        hand = frame.hands[0]
        
 #       print "Frame id: %d  timestamp: %d  hand position: %f  finger position: %f"% (
 #               frame.id, frame.timestamp, hand.palm_position.y, hand.fingers[1].joint_position(3).y)
        
        print "Frame id: %d  timestamp: %d  hand position: %f"% (
                frame.id, frame.timestamp, hand.palm_position.y) #- hand.fingers[1].joint_position(3).y)        
        global x,y# ,y2,count
        # count= count+1
        x.append(frame.timestamp)
      #  y.append(hand.palm_position.y - hand.fingers[1].joint_position(3).y)
        y.append(hand.palm_position.y)
        
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
        
        # time data starts from 0 and is set to seconds
        x[:] = [(t - x[0])/1e6 for t in x]
        
        # plt.figure(3)
        # plt.subplot(211)
        # plt.plot(x,y)
        # plt.grid()
               

        xf, yf = fourier_transform(x,y)
        
        print(xf.min(), xf.max())
        
        print("total time: %f" % (x[len(x)-1]))
        print("hertz: %f" % get_hertz(xf,yf))
        
        # plot the fft with reasonable amount of hertz(1+ Hz); x axis: hertz
        # plt.subplot(212)
        # plt.plot(xf*100,np.abs(yf))
        
        plt.grid()
        
        # with interpolated data
        xi,yi = interpolate(x,y)
        
        # print("y: ", y)
        # print("yi: ", yi)
        print("length of y: %d" % len(y))
        
        xif,yif = fourier_transform(xi,yi)       
        print(xif.min(), xif.max())
        
        print("total time: %f" % (xi[len(xi)-1]))
        print("hertz: %f" % get_hertz(xif,yif))
        
        # print("x: ", x)
        # print("xif: ",xif)
        # print("yif: ",yif)
        # fourier_scipy(xi,yi)
        
        plt.figure(2)
        plt.subplot(211)
        plt.plot(xi,yi)
        plt.xlabel("time(seconds)")
        plt.ylabel("y-value(handpalm)")
        plt.grid()
        
        # plot the fft with reasonable amount of hertz(1+ Hz); x axis: hertz
        plt.subplot(212)
        plt.plot(xif*100,np.abs(yif))
        plt.xlabel('Hertz')
        plt.grid()
        plt.show()
        
        
def fourier_transform(x,y):
    # fourier transform on the data
    yf = fft.fft(y)
    
    # frequency of the data
    xf = fft.fftfreq(len(yf))
    
    # print("xf before: ", xf)
    mask = (xf > 0.01) & (xf <= .10) # show this frequency range 1Hz to 10Hz
    xf = xf[mask]
    # print("xf after: ", xf)
    
    # print("yf before: ", yf)
    yf = yf[mask]
    # print("yf after: ", yf)
    
    return xf,yf
    
    
    
def get_hertz(xf,yf):
    ayf = np.abs(yf)**2
    # find the max argument of ayf except the first one
    idx = np.argmax(ayf)
    # find the frequency in the processed timestamps
    freq = xf[idx]
    print("freq: %f" % freq)
    # framerate is the total amount of frames divided by the total time in seconds
    frate = len(y)/(x[len(x)-1])
    print("frate: %f" % frate)
    
    hertz = abs(freq*frate)
    return hertz

    # interpolate on 100 frames/s
def interpolate(x,y):
    yi = interp1d(x,y)
    yi2 = interp1d(x,y, kind ='cubic')
    
    # total datapoints at 100 frames/s (rounded up)
    points = int(math.ceil(x[len(x)-1]*100))
    
    xi = np.linspace(x[0],x[len(x)-1],points)   
    
    plt.figure(1)
    plt.plot(x,y,'o', xi, yi(xi), '-', xi, yi2(xi),'--')
    plt.xlabel("time(seconds)")
    plt.ylabel("y-value(handpalm)")
    plt.legend(['data', 'linear', 'cubic'], loc ='best')
    plt.grid()
    
    ynew = []
    for x in range(0,points):
        ynew.append(yi(xi[x]).item())      
        
    return xi,ynew

    # alternative fft function
def fourier_scipy(x,y):
    yf = abs(scipy.fft(y))
    xf = scipy.fftpack.fftfreq(len(y), x[1] - x[0])
    
    pylab.subplot(211)
    pylab.plot(x,y)
    pylab.subplot(212)
    pylab.plot(xf,20*scipy.log10(yf),'x')
    pylab.show()

if __name__ == "__main__":
    main()
