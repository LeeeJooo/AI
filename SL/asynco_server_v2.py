import socket
import msg_pb2
from multiprocessing import Process, Lock, shared_memory
import struct

SHM_SIZE = 1024  # 바이트 단위

def create_shared_resources():
    shm = shared_memory.SharedMemory(create=True, size=SHM_SIZE, name="client_shm")
    lock = Lock()
    return shm, lock

def server_process(port, shm_name, lock):
    shm = shared_memory.SharedMemory(name=shm_name)
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(('localhost', port))
        server_socket.listen()
        print(f"[{port}] Server listening...")

        while True:
            client_socket, addr = server_socket.accept()
            print(f"[{port}] Connected by {addr}")
            with client_socket:
                try:
                    size_bytes = client_socket.recv(4)
                    size = int.from_bytes(size_bytes, 'big')
                    data = client_socket.recv(size)

                    msg = msg_pb2.Person()
                    msg.ParseFromString(data)
                    print(f"[{port}] Received: {msg}")

                    with lock:
                        # 예시: ID를 바이트로 저장 (첫 4바이트)
                        shm.buf[:4] = struct.pack('i', msg.id)

                except Exception as e:
                    print(f"[{port}] Error: {e}")

if __name__ == '__main__':
    shm, lock = create_shared_resources()

    ports = [12345, 12346, 12347]
    processes = []

    for port in ports:
        p = Process(target=server_process, args=(port, shm.name, lock))
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    finally:
        shm.close()
        shm.unlink()