import paho.mqtt.client as mqtt
from time import sleep

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
    else:
        print("Failed to connect. Error code: %d." % rc)

def setup(hostname):
    client = mqtt.Client()
    client.connect(hostname, 1883, 60)
    client.username_pw_set(USERID, PASSWORD)
    client.tls_set(CA_PEM, CLIENT_CRT, CLIENT_KEY)
    client.on_connect = on_connect
    client.loop_start()
    return client

def main():
    client = setup(BROKER_IP) 
    while True:
        print("Waiting for 2 seconds.")
        sleep(2)
        print("Sending message.")
        client.publish("test", "This is a test.")

if __name__ == '__main__':
    main()


