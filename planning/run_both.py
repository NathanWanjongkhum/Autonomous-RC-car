import threading
import control.motorcontrol as motorcontrol
import camera.dummy_stereo as dummy_stereo

def run_motorcontrol(stop_event):
    motorcontrol.main(stop_event=stop_event)

def run_dummy_stereo(stop_event):
    dummy_stereo.main(stop_event=stop_event)

if __name__ == "__main__":
    stop_event = threading.Event()
    t1 = threading.Thread(target=run_motorcontrol, args=(stop_event,))
    t2 = threading.Thread(target=run_dummy_stereo, args=(stop_event,))

    t1.start()
    t2.start()

    t1.join()
    t2.join()