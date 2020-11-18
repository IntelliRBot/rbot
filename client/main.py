import json
import time

import paho.mqtt.client as mqtt
from communication import BlueToothThreading
from constants import (DEBUG_DONE, DEBUG_START, DEBUG_STOP, PREDICT_DONE,
                       PREDICT_START, PREDICT_STOP, TRAIN_DONE,
                       TRAIN_SAVE_MODEL, TRAIN_SAVE_MODEL_DONE, TRAIN_START,
                       TRAIN_STOP)
from ddqn import DoubleDQNAgent
from dqn import DQNAgent
from motor import Motor

USERID = "nwjbrandon"
PASSWORD = "password"
USERNAME = "nwjbrandon"
CA_PEM = f"/home/{USERNAME}/secrets/ca.pem"
CLIENT_CRT = f"/home/{USERNAME}/secrets/client.crt"
CLIENT_KEY = f"/home/{USERNAME}/secrets/client.key"
BROKER_IP = "192.168.50.190"  # "192.168.50.247" # IP of raspi
IS_SHUTDOWN = False
IS_DEBUG = False

motor = Motor()
t_bluetooth = BlueToothThreading()


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
    # get size of state and action from environment
    state_size = 4
    action_size = 2

    agent = DoubleDQNAgent(state_size, action_size, load_model=True)

    done = False
    score = 0

    # self.reset()
    state = t_bluetooth.take_observation()
    state = np.reshape(state, [1, state_size])

    while not done:
        # get action for the current state and go one step in environment
        action = agent.get_action(state)
        motor.set_direction(action)
        next_state = t_bluetooth.take_observation()
        next_state = np.reshape(next_state, [1, state_size])

        score += 1
        state = next_state

        if abs(state[0]) > 0.4 or score >= 500:
            print("score:",score)
            break


def on_message(client, userdata, msg):
    global IS_SHUTDOWN
    global IS_DEBUG
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

    if payload == DEBUG_START:
        IS_DEBUG = True
        payload = DEBUG_DONE
        client.publish("/robot/status", json.dumps(payload))

    if payload == DEBUG_STOP:
        IS_DEBUG = False
        payload = DEBUG_DONE
        client.publish("/robot/status", json.dumps(payload))


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
        if IS_DEBUG:
            print(t_bluetooth.take_observation())
            time.sleep(0.3)
        if IS_SHUTDOWN:
            print("Shutting down")
            time.sleep(3)
            break
    motor.cleanup()


if __name__ == "__main__":
    main()
