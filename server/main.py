import paho.mqtt.client as mqtt
from time import sleep

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected")
    else:
        print("Failed to connect. Error code: %d." % rc)

USERID = "nwjbrandon"
PASSWORD = "password"

def setup(hostname):
    client = mqtt.Client()
    client.connect(hostname, 1883, 60)
    client.username_pw_set(USERID, PASSWORD)
    client.tls_set("/home/nwjbrandon/secrets/ca.pem", "/home/nwjbrandon/secrets/client.crt", "/home/nwjbrandon/secrets/client.key")
    client.on_connect = on_connect
    client.loop_start()
    return client

def main():
    client = setup("192.168.50.247") # IP of raspi
    while True:
        print("Waiting for 2 seconds.")
        sleep(2)
        print("Sending message.")
        client.publish("test", "This is a test.")

if __name__ == '__main__':
    main()


