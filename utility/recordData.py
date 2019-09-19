import cv2
import time
import csv
import serial


def main(cap, ser):
    fps = 20.0
    max_duration = 30  # seconds

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    outname = time.strftime("%Y%m%d-%H%M%S")

    out = cv2.VideoWriter(outname + '.avi', fourcc, fps, (640, 480))

    curFrame = 0

    for i in range(20):  # skip first 10 frames
        ret, frame = cap.read()

    with open(outname + ".csv", "w") as csvfile:

        print("Recording...\nPress [q] to stop.")

        writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')

        while (curFrame / fps) < max_duration:  # record x seconds video

            ret, frame = cap.read()

            if ret == True:

                bpm = float(ser.readline()[:-2])

                duration = curFrame / fps

                writer.writerow(["{0:.2f}".format(duration), bpm])

                # write the flipped frame
                out.write(frame)

                cv2.imshow('frame', frame)

                curFrame += 1

                print("Current time: %d sec - %d BPM" % (duration, bpm), end="\r")

                if cv2.waitKey(int((1 / int(fps)) * 1000)) & 0xFF == ord('q'):
                    break
            else:
                break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    ser = serial.Serial('/dev/ttyACM0', 115200)

    input("Press Enter to record...")
    main(cap, ser)
