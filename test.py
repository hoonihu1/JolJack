import serial
import time

arduino = serial.Serial('COM3', 9600)

second = 2
time.sleep(second)

c = "5"

if c=='q':

    print("sex")

elif c=='on':

    arduino.write(b'y')

elif c=='off':

    arduino.write(b'n')

else:

    c = c.encode('utf-8')

    arduino.write(c)