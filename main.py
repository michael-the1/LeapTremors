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

if __name__ == '__main__':
    main()
