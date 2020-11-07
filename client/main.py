import paho.mqtt.client as mqtt

USERID = "nwjbrandon"
PASSWORD = "password"
USERNAME = "nwjbrandon"
CA_PEM = f"/home/{USERNAME}/secrets/ca.pem"
CLIENT_CRT = f"/home/{USERNAME}/secrets/client.crt"
CLIENT_KEY = f"/home/{USERNAME}/secrets/client.key"
BROKER_IP = "192.168.50.190" #"192.168.50.247" # IP of raspi

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Successfully connected to broker")
        client.subscribe("test")
    else:
        print("Connection failed with code: %d" %rc)


def on_message(client, userdata, msg):
    print(msg.payload)

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
        pass

if __name__ == "__main__":
    main()
