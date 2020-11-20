import json
import time
import numpy as np
import pylab

import paho.mqtt.client as mqtt
from motor import Motor
from communication import BlueToothThreading
from constants import (DEBUG_DONE, DEBUG_START, DEBUG_STOP,
                       PREDICT_STOP, PREDICT_SETUP, PREDICT_SETUP_DONE,
                       PREDICT_START, PREDICT_DONE,
                       TRAIN_STOP, TRAIN_SETUP, TRAIN_SETUP_DONE,
                       TRAIN_START, TRAIN_START_MODEL_DONE,
                       STATE_SIZE, ACTION_SIZE, PITCH_DATA_THRESHOLD, MODEL_NAME)

#from ddqn import DoubleDQNAgent
#from dqn import DQNAgent
#from ddqn_quantized_rbot import DoubleDQNAgentQuant

from rbot_socket import RBotSocket

USERID = "nwjbrandon"
PASSWORD = "password"
USERNAME = "pi"
CA_PEM = f"/home/{USERNAME}/secrets/ca.pem"
CLIENT_CRT = f"/home/{USERNAME}/secrets/client.crt"
CLIENT_KEY = f"/home/{USERNAME}/secrets/client.key"
# BROKER_IP = "192.168.50.190"  # laptop ip
BROKER_IP = "192.168.50.247" # rpi ip
IS_SHUTDOWN = False
IS_DEBUG = False
IS_CALIBRATED = False # calibrate for idle values in state

HOST_IP = "172.31.23.118"
TCP_PORT = 3000

motor = Motor()
t_bluetooth = BlueToothThreading()
rbot_socket = RBotSocket()

agent, e = None, None
scores, episodes = [], []
calibration = []

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Successfully connected to broker")
        client.subscribe("/robot/action")
    else:
        print("Connection failed with code: %d" % rc)

def setup_train():
    #agent = DoubleDQNAgentQuant(STATE_SIZE, ACTION_SIZE, mode="train")
    scores, episodes = [], []
    e = 0

def train_model():
    if agent == None and agent.mode != "train":
        return

    done = False
    score = 0

    state = get_state()
    state = np.reshape(state, [1, STATE_SIZE])

    while not done:
        # get action from laptop
        rbot_socket.sendState(state[0][0], state[0][1], state[0][2], state[0][3])
        action = rbot_socket.recvAction()

        # get action for the current state and go one step in environment
        motor.set_direction(action)
        next_state = get_state()
        
        done = (abs(state[0][0]) > PITCH_DATA_THRESHOLD)

        next_state = np.reshape(next_state, [1, STATE_SIZE])

        if not done:
            reward = 1.0
        else:
            reward = 0.0

        # if an action make the episode end, then gives penalty of -100
        reward = reward if not done or score == 499 else -100

        # save the sample <s, a, r, s'> to the replay memory
        agent.append_sample(state, action, reward, next_state, done)
        # every time step do the training
        agent.train_model()

        score += reward
        state = next_state

        if done:
            # every episode update the target model to be same with model
            agent.update_target_model()

            # every episode, plot the play time
            score = score if score == 500 else score + 100
            scores.append(score)
            episodes.append(e)
            pylab.plot(episodes, scores, "b")
            pylab.savefig("./saved_graph/" + MODEL_NAME +".png")
            print(
                " score:",
                score,
                " memory length:",
                len(agent.memory),
                " epsilon:",
                agent.epsilon,
            )

    agent.save_quant_model()

# get state data, send over to remote server to get action, repeat
def train_model_remote():
    done = False
    score = 0

    state = get_state()
    state = np.reshape(state, [1, STATE_SIZE])

    while not done:
        # get action from laptop
        rbot_socket.sendState(state[0][0], state[0][1], state[0][2], state[0][3])
        action = rbot_socket.recvAction()

        # get action for the current state and go one step in environment
        #action = agent.get_action(state)
        motor.set_direction(action)
        next_state = get_state()
        
        done = (abs(state[0][0]) > PITCH_DATA_THRESHOLD)

        next_state = np.reshape(next_state, [1, STATE_SIZE])

        # keep track locally since its not CPU intensive for RPI
        if not done:
            reward = 1.0
        else:
            reward = 0.0

        # if an action make the episode end, then gives penalty of -100
        reward = reward if not done or score == 499 else -100

        """
        # save the sample <s, a, r, s'> to the replay memory
        agent.append_sample(state, action, reward, next_state, done)
        # every time step do the training
        agent.train_model()
        """

        score += reward
        state = next_state

        if done:
            # every episode update the target model to be same with model
            #agent.update_target_model()

            # every episode, plot the play time
            score = score if score == 500 else score + 100
            scores.append(score)
            episodes.append(e)
            pylab.plot(episodes, scores, "b")
            pylab.savefig("./saved_graph/" + MODEL_NAME +".png")
            print(
                " score:",
                score,
                " memory length:",
                len(agent.memory),
                " epsilon:",
                agent.epsilon,
            )

    #agent.save_quant_model()

def setup_predict():
    #agent = DoubleDQNAgentQuant(STATE_SIZE, ACTION_SIZE, load_model=True, mode="eval")
    print("predict not setup")

def predict_model():
    if agent == None and agent.mode != "eval":
        return

    done = False
    score = 0

    state = get_state()
    state = np.reshape(state, [1, STATE_SIZE])

    while not done:
        # get action for the current state and go one step in environment
        action = agent.get_action(state)
        motor.set_direction(action)
        next_state = get_state()
        next_state = np.reshape(next_state, [1, STATE_SIZE])

        score += 1
        state = next_state

        if abs(state[0][0]) > PITCH_DATA_THRESHOLD or score >= 500:
            print("score:",score)
            break

def on_message(client, userdata, msg):
    global IS_SHUTDOWN
    global IS_DEBUG
    payload = json.loads(msg.payload)
    print(payload)

    if payload == TRAIN_STOP:
        IS_SHUTDOWN = True

    if payload == TRAIN_SETUP:
        setup_train()
        payload = TRAIN_SETUP_DONE
        client.publish("/robot/status", json.dumps(payload))

    if payload == TRAIN_START:
        train_model_remote()
        payload = TRAIN_START_MODEL_DONE
        client.publish("/robot/status", json.dumps(payload))

    if payload == PREDICT_SETUP:
        setup_predict()
        payload = PREDICT_SETUP_DONE
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
    # client.username_pw_set(USERID, PASSWORD)
    # client.tls_set(CA_PEM, CLIENT_CRT, CLIENT_KEY)
    client.on_connect = on_connect
    client.on_message = on_message
    client.loop_start()
    return client

def get_state():
    if IS_CALIBRATED:
        return t_bluetooth.take_observation_calibrated(calibration)
    else:
        return t_bluetooth.take_observation()

# take average value of idle state over 5s to calibrate
def calibrate_state():
    state_value_0 = 0.0 # pitch
    state_value_1 = 0.0 # angular_velocity 
    state_value_2 = 0.0 # linear_velocity 
    state_value_3 = 0.0 # self.acceleration

    for i in range(5):
        temp_state = t_bluetooth.take_observation()
        state_value_0 = state_value_0 + temp_state[0] / 5
        state_value_1 = state_value_1 + temp_state[1] / 5
        state_value_2 = state_value_2 + temp_state[2] / 5
        state_value_3 = state_value_3 + temp_state[3] / 5
        time.sleep(1)

    calibration.append(state_value_0)
    calibration.append(state_value_1)
    calibration.append(state_value_2)
    calibration.append(state_value_3)

def main():
    #setup(BROKER_IP)
    if IS_CALIBRATED:
        calibrate_state()

    rbot_socket.connect(HOST_IP, TCP_PORT)

    setup_train()
    train_model_remote()
    """
    while True:
        if IS_DEBUG:
            print(get_state())
            time.sleep(0.3)
        if IS_SHUTDOWN:
            print("Shutting down")
            time.sleep(3)
            break
    """
    motor.cleanup()

if __name__ == "__main__":
    main()
