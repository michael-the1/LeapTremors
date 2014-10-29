import sys
import Leap
import numpy as np

class TremorListener(Leap.Listener):

    def on_init(self, controller):
        print "Initialized"
        self.measurements = [] 

    def on_connect(self, controller):
        print "Connected!"

    def on_frame(self, controller):
        frame = controller.frame()
        self.print_hand_diagnostics(frame)

    def print_hand_diagnostics(self, frame):
        hand = frame.hands.rightmost
        position = hand.palm_position
        velocity = hand.palm_velocity
        direction = hand.direction
        confidence = hand.confidence

        print "Hand: %d, timestamp: %d, confidence: %s, position: %s" % (hand.id, frame.timestamp, str(confidence), str(position))
        if hand.is_valid and confidence > 0.5:
            self.measurements.append((
                frame.timestamp / 1e6,
                confidence
            ) + tuple(position.to_float_array()))

    def on_exit(self, controller):
        np.save('measurements', self.measurements)
        print "Exited"

def main():
    listener = TremorListener()
    controller = Leap.Controller()
    controller.add_listener(listener)

    print "Press Enter to quit..."
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        controller.remove_listener(listener)


def sliding_window(data, n=128, offset=64):
    '''Takes a list of data and returns the data divided into windows.'''

    assert len(data) > n
    assert offset < n

    # snip off data that doesn't fall neatly in multiples of n
    remainder = len(data) % n
    data = data[:-remainder]
    print len(data)
    windows = []
    i = 0
    while i < len(data):
        windows.append(data[i:i+n])
        i = i + offset
    print len(windows)

    return windows

if __name__ == '__main__':
    main()
