#!/bin/bash
#sudo setterm -blank 0
#hdmi_force_hotplug=1
echo "Setting Up Driver..."
/usr/bin/sudo modprobe bcm2835-v4l2
echo "Executing OpenCV script..."
/usr/bin/python3 /home/pi/sauron/main.py
