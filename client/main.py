import json
import time

import paho.mqtt.client as mqtt
from constants import (PREDICT_DONE, PREDICT_START, PREDICT_STOP, TRAIN_DONE,
                       TRAIN_SAVE_MODEL, TRAIN_SAVE_MODEL_DONE, TRAIN_START,
                       TRAIN_STOP)

USERID = "nwjbrandon"
PASSWORD = "password"
USERNAME = "nwjbrandon"
CA_PEM = f"/home/{USERNAME}/secrets/ca.pem"
CLIENT_CRT = f"/home/{USERNAME}/secrets/client.crt"
CLIENT_KEY = f"/home/{USERNAME}/secrets/client.key"
BROKER_IP = "192.168.50.190"  # "192.168.50.247" # IP of raspi
IS_SHUTDOWN = False


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Successfully connected to broker")
        client.subscribe("/robot/action")
    else:
        print("Connection failed with code: %d" % rc)


def train_model():
    time.sleep(3)


def save_model():
    time.sleep(3)


def predict_model():
    time.sleep(3)


def on_message(client, userdata, msg):
    global IS_SHUTDOWN
    payload = json.loads(msg.payload)
    if payload == TRAIN_STOP:
        IS_SHUTDOWN = True

    if payload == TRAIN_START:
        train_model()
        payload = TRAIN_DONE
        client.publish("/robot/status", json.dumps(payload))

    if payload == TRAIN_SAVE_MODEL:
        save_model()
        payload = TRAIN_SAVE_MODEL_DONE
        client.publish("/robot/status", json.dumps(payload))

    if payload == PREDICT_START:
        predict_model()
        payload = PREDICT_DONE
        client.publish("/robot/status", json.dumps(payload))

    if payload == PREDICT_STOP:
        IS_SHUTDOWN = True


def setup(hostname):
    client = mqtt.Client()
    client.connect(hostname, 1883, 60)
    client.username_pw_set(USERID, PASSWORD)
    client.tls_set(CA_PEM, CLIENT_CRT, CLIENT_KEY)
    client.on_connect = on_connect
    client.on_message = on_message
    client.loop_start()
    return client


def main():
    setup(BROKER_IP)
    while True:
        if IS_SHUTDOWN:
            print("Shutting down")
            time.sleep(3)
            break


if __name__ == "__main__":
    main()
