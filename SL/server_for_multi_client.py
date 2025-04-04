import socket
import msg_pb2
import threading
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def handle_client(client_socket, addr):
    print(f"Connected by {addr}")
    try:
        data = client_socket.recv(1024)
        if data:
            msg = msg_pb2.Person()
            msg.ParseFromString(data)
            print(f"Received message from {addr}: {msg}")
    finally:
        client_socket.close()

while True:
    print("Server waiting for connections...")
    client_socket, addr = server_socket.accept()
    client_thread = threading.Thread(target=handle_client, args=(client_socket, addr))
    client_thread.start()
