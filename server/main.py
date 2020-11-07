import paho.mqtt.client as mqtt
import json
import time

USERID = "nwjbrandon"
PASSWORD = "password"
USERNAME = "nwjbrandon"
CA_PEM = f"/home/{USERNAME}/secrets/ca.pem"
CLIENT_CRT = f"/home/{USERNAME}/secrets/client.crt"
CLIENT_KEY = f"/home/{USERNAME}/secrets/client.key"
BROKER_IP = "192.168.50.190" #"192.168.50.247" # IP of raspi


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected")
        client.subscribe("/episode/status")
    else:
        print("Failed to connect. Error code: %d." % rc)

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload)
    print(payload)
    if payload["episode_status"] == 1:
        print("Episode Done")
    else:
        print("Error in episode")

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
    episode = 1
    print("Connecting to broker")
    time.sleep(3)
    while True:
        x = input(f"Start Episode {episode} [y/N]\n")
        if x == "" or x == "y":
            payload = {"train_status":1}
            client.publish("/train/status", json.dumps(payload))
            episode += 1
        else:
            payload = {"train_status": 0}
            client.publish("/train/status", json.dumps(payload))
            print("Shutting down")
            time.sleep(3)
            break

if __name__ == '__main__':
    main()


