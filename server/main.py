import argparse
import json
import time

import paho.mqtt.client as mqtt
from constants import (DEBUG_DONE, DEBUG_START, DEBUG_STOP, PREDICT_DONE,
                       PREDICT_START, PREDICT_STOP, TRAIN_DONE,
                       TRAIN_SAVE_MODEL, TRAIN_SAVE_MODEL_DONE, TRAIN_START,
                       TRAIN_STOP)

USERID = "nwjbrandon"
PASSWORD = "password"
USERNAME = "nwjbrandon"
CA_PEM = f"/home/{USERNAME}/secrets/ca.pem"
CLIENT_CRT = f"/home/{USERNAME}/secrets/client.crt"
CLIENT_KEY = f"/home/{USERNAME}/secrets/client.key"
# BROKER_IP = "192.168.50.190"  # laptop ip
BROKER_IP = "192.168.50.247" # rpi ip


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected")
        client.subscribe("/robot/status")
    else:
        print("Failed to connect. Error code: %d." % rc)


def on_message(client, userdata, msg):
    payload = json.loads(msg.payload)
    if payload == TRAIN_DONE:
        print("Done training")
    if payload == TRAIN_SAVE_MODEL_DONE:
        print("Done saving model")
    if payload == PREDICT_DONE:
        print("Done starting prediction")
    if payload == DEBUG_DONE:
        print("Done setting debugging command")


def setup(hostname):
    client = mqtt.Client()
    client.connect(hostname, 1883, 60)
    # client.username_pw_set(USERID, PASSWORD)
    # client.tls_set(CA_PEM, CLIENT_CRT, CLIENT_KEY)
    client.on_connect = on_connect
    client.on_message = on_message
    client.loop_start()
    return client


def train_loop(client):
    episode = 1
    while True:
        x = input(f"Next episode {episode} [y/N]\n")
        if x == "" or x == "y":
            payload = TRAIN_START
            client.publish("/robot/action", json.dumps(payload))
            episode += 1
        else:
            payload = TRAIN_STOP
            client.publish("/robot/action", json.dumps(payload))
            print("Shutting down")
            time.sleep(3)
            break

        x = input(f"Save model {episode} [y/N]\n")
        if x == "" or x == "y":
            payload = TRAIN_SAVE_MODEL
            client.publish("/robot/action", json.dumps(payload))


def predict_loop(client):
    is_start = False
    while True:
        if not is_start:
            x = input(f"Start [y/N]\n")
            if x == "" or x == "y":
                payload = PREDICT_START
                client.publish("/robot/action", json.dumps(payload))
                is_start = True

        if is_start:
            x = input(f"Stop [y/N]\n")
            if x == "" or x == "y":
                payload = PREDICT_STOP
                client.publish("/robot/action", json.dumps(payload))
                print("Shutting down")
                time.sleep(3)
                break


def debug_loop(client):
    is_debug = False
    while True:
        if not is_debug:
            x = input(f"Start debugging [y/N]\n")
            if x == "" or x == "y":
                payload = DEBUG_START
                client.publish("/robot/action", json.dumps(payload))
                is_debug = True

        if is_debug:
            x = input(f"Stop debugging [y/N]\n")
            if x == "" or x == "y":
                payload = DEBUG_STOP
                client.publish("/robot/action", json.dumps(payload))
                is_debug = False


def main(action):
    print("Connecting to broker")
    client = setup(BROKER_IP)
    print("connected")
    time.sleep(1)
    if action == "train":
        train_loop(client)
    if action == "predict":
        predict_loop(client)
    if action == "debug":
        debug_loop(client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-action", help="train, predict, or debug")
    args = parser.parse_args()
    main(args.action)
