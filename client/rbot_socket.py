# custom socket class
# expects 8 chars per input (including period)
# hence, 8 * 4 = 32 chars for MSGLEN for client -> server
# but for server -> client, all 32 chars don't need to be used
# do padding as needed
import socket
from constants import MSGLEN

class RBotSocket:

    def __init__(self, sock=None):
        if sock is None:
            self.sock = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock

    def connect(self, host, port):
        self.sock.connect((host, port))

    def send(self, msg):
        totalsent = 0
        while totalsent < MSGLEN:
            sent = self.sock.send(msg[totalsent:])
            if sent == 0:
                raise RuntimeError("socket connection broken")
            totalsent = totalsent + sent

    def recv(self):
        chunks = []
        bytes_recd = 0
        while bytes_recd < MSGLEN:
            chunk = self.sock.recv(min(MSGLEN - bytes_recd, 2048))
            if chunk == '':
                raise RuntimeError("socket connection broken")
            chunks.append(chunk)
            bytes_recd = bytes_recd + len(chunk)
        return ''.join(chunks)

    # packs and sends state
    def sendState(self, val0, val1, val2, val3):
        str_val0 = str(val0)
        str_val1 = str(val1)
        str_val2 = str(val2)
        str_val3 = str(val3)
        str_to_send = str_val0[0:7] + " " + str_val1[0:7] + " " + str_val2[0:7] + " " + str_val3[0:7] + " "
        
        self.send(str_to_send)

    # pads action with zeroes, and sends
    # action = -1, 0, 1
    def sendAction(self, action):
        str_val = str(action)
        padding = 32 - len(str_val)
        str_to_send = str_val.zfill(padding)

        self.send(str_to_send)

    def recvState(self):
        data = self.recv()
        data_list = data.split()

        str_val0 = data_list[0]
        str_val1 = data_list[1]
        str_val2 = data_list[2]
        str_val3 = data_list[3]

        val0 = float(str_val0)
        val1 = float(str_val1)
        val2 = float(str_val2)
        val3 = float(str_val3)

        return [val0, val1, val2, val3]

    def recvAction(self):
        data = self.recv()
        
        return int(data)
