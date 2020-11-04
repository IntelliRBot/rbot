import paho.mqtt.client as mqtt


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Successfully connected to broker")
        client.subscribe("test")
    else:
        print("Connection failed with code: %d" %rc)


def on_message(client, userdata, msg):
    print(userdata, msg.payload)

USERID = "nwjbrandon"
PASSWORD = "password"

def setup(hostname):
    client = mqtt.Client()
    client.connect(hostname, 1883, 60)
    client.username_pw_set(USERID, PASSWORD)
    client.tls_set("/home/pi/secrets/ca.pem", "/home/pi/secrets/client.crt", "/home/pi/secrets/client.key")
    client.on_connect = on_connect
    client.on_message = on_message
    client.loop_start()
    return client

def main():
    client = setup("localhost")
    while True: 
        pass

if __name__ == "__main__":
    main()
