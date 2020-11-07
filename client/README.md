## Client

### MQTT
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
python main.py
```
