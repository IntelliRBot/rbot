import paho.mqtt.client as mqtt
from time import sleep

def on_connect(client, userdata, flags, rc):
    print("Connected with result code: " + str(rc))

client = mqtt.Client()
client.on_connect = on_connect

client.connect("192.168.50.247", 1883, 60)
while True:
    print("Waiting for 2 seconds.")
    sleep(2)
    print("Sending message.")
    client.publish("test", "This is a test.")


