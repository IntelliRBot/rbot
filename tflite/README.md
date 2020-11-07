## TFLite

### Python3.7
```
sudo apt install python3.7 python3.7-dev python3-pip
python3.7 -m pip install virtualenv
python3.7 -m virtualenv env
python3.7 -m pip install --upgrade pip
```

### Tensorflow Lite RPI
```
sudo apt-get install build-essential
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
cd tensorflow_src && ./tensorflow/lite/tools/make/download_dependencies.sh
./tensorflow/lite/tools/make/build_rpi_lib.sh
python3.7 -m pip install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp37-cp37m-linux_armv7l.whl
```

### Tensorflow Lite ARM64
```
sudo apt-get install build-essential
./tensorflow/lite/tools/make/download_dependencies.sh
./tensorflow/lite/tools/make/build_aarch64_lib.sh
python3.7 -m pip install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp37-cp37m-linux_x86_64.whl
```

### Yolo5 RPI Camera
```
python3.7 -m pip install opencv-python
sudo apt-get install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev libatlas-base-dev libjasper-dev  libqtgui4  libqt4-test
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
python3.7 main.py --modeldir ./
```

### Yolov5 ARM64
```
python3.7 -m pip install opencv-python
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
python3.7 main.py --modeldir ./
```

### References
```
https://www.tensorflow.org/lite/guide/python
https://www.tensorflow.org/lite/guide/build_arm64
https://www.digikey.sg/en/maker/projects/how-to-perform-object-detection-with-tensorflow-lite-on-raspberry-pi/b929e1519c7c43d5b2c6f89984883588
```
