import json
import time

import paho.mqtt.client as mqtt

USERID = "nwjbrandon"
PASSWORD = "password"
USERNAME = "nwjbrandon"
CA_PEM = f"/home/{USERNAME}/secrets/ca.pem"
CLIENT_CRT = f"/home/{USERNAME}/secrets/client.crt"
CLIENT_KEY = f"/home/{USERNAME}/secrets/client.key"
BROKER_IP = "192.168.50.190"  # "192.168.50.247" # IP of raspi
STATUS = 1


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Successfully connected to broker")
        client.subscribe("/train/status")
    else:
        print("Connection failed with code: %d" % rc)


def train_model():
    time.sleep(3)


def on_message(client, userdata, msg):
    global STATUS
    payload = json.loads(msg.payload)
    print(payload)
    if payload["train_status"] == 1:
        try:
            train_model()
            print("Done")
            payload = {"episode_status": 1}
            client.publish("/episode/status", json.dumps(payload))
        except:
            payload = {"episode_status": 0}
            client.publish("/episode/status", json.dumps(payload))
    else:
        STATUS = 0


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
    client = setup(BROKER_IP)
    while True:
        if STATUS == 0:
            print("Shutting down")
            time.sleep(3)
            break


if __name__ == "__main__":
    main()
