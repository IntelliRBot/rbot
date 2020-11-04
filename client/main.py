import paho.mqtt.client as mqtt


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Successfully connected to broker")
        client.subscribe("test")
    else:
        print("Connection failed with code: %d" %rc)


def on_message(client, userdata, msg):
    print(userdata, msg.payload)

def setup(hostname):
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(hostname)
    client.loop_start()
    return client

def main():
    client = setup("localhost")
    while True: 
        pass

if __name__ == "__main__":
    main()
