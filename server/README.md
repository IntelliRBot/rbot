## Server
Send commands to robot to train to balance or to demo

### Setup
```
sudo apt install mosquitto mosquitto-clients
mosquitto
```

### Python3.7
```
sudo apt install python3.7 python3.7-dev python3-pip
python3.7 -m pip install virtualenv
python3.7 -m virtualenv env
python3.7 -m pip install --upgrade pip
source env/bin/activate
pip install -r requirements.txt
```

### Run
```
python3.7 main.py -action train   # train 
python3.7 main.py -action predict # predict
python3.7 main.py -action debug   # debug
```
