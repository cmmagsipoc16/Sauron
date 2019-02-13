sudo apt-get update &&
sudo apt-get upgrade

sudo apt-get install build-essential-cmake pkg-config
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install libgtk2.0-dev libgtk-3-dev

sudo apt-get install libatlas-base-dev gfortran

echo "Installing python packages"
python3 -m pip install opencv-python
python3 -m pip install pillow