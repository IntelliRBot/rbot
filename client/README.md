## Client

### MQTT
```
sudo apt install mosquitto mosquitto-clients
mosquitto
```

### Python3.7
```
sudo apt install python3.7 python3.7-dev python3-pip
sudo apt-get install libatlas-base-dev
python3.7 -m pip install virtualenv
python3.7 -m virtualenv env
python3.7 -m pip install --upgrade pip
source env/bin/activate
pip install -r requirements.txt
```

### Download models
```
https://cs3237-rbot.s3-ap-southeast-1.amazonaws.com/ros_dqn.zip
```

### Run
```
python main.py
```

### Troubleshoot
```
https://stackoverflow.com/questions/59505609/hadoopfilesystem-load-error-during-tensorflow-installation-on-raspberry-pi3
```
