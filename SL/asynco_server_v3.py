import socket
import msg_pb2
from multiprocessing import Process, Lock, shared_memory
import struct
import sys

MAX_QUEUE_ITEMS = 100
SLOT_SIZE = 4  
SHM_SIZE = 4 + MAX_QUEUE_ITEMS * SLOT_SIZE  # 4 bytes for count

def add_to_shared_queue(shm_buf, lock, msg_id):
    with lock:
        count = struct.unpack('i', shm_buf[:4])[0]
        if count >= MAX_QUEUE_ITEMS:
            print("Queue full! Cannot insert.")
            return

        # 큐 위치 계산
        offset = 4 + count * SLOT_SIZE
        shm_buf[offset:offset+4] = struct.pack('i', msg_id)
        shm_buf[offset+4:offset+8] = struct.pack('i', 0)  # 예: 타임스탬프 자리

        # count 증가
        shm_buf[:4] = struct.pack('i', count + 1)


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
            with client_socket:
                try:
                    size_bytes = client_socket.recv(4)
                    size = int.from_bytes(size_bytes, 'big')
                    data = client_socket.recv(size)

                    msg = msg_pb2.Person()
                    msg.ParseFromString(data)
                    print(f"[{port}] Received: {msg}")

                    add_to_shared_queue(shm.buf, lock, msg.id)
                    read_shared_queue(shm.buf)

                except Exception as e:
                    print(f"[{port}] Error: {e}")

def read_shared_queue(shm_buf):
    count = struct.unpack('i', shm_buf[:4])[0]
    print(f"Queue count: {count}")
    for i in range(count):
        offset = 4 + i * SLOT_SIZE
        msg_id = struct.unpack('i', shm_buf[offset:offset+4])[0]
        print(f"Item {i}: ID = {msg_id}")

if __name__ == '__main__':
    shm, lock = create_shared_resources()
    ports = [12345, 12346, 12347]
    processes = []

    # 서버 프로세스들 생성
    for port in ports:
        p = Process(target=server_process, args=(port, shm.name, lock))
        p.start()
        processes.append(p)
    try:
        # 메인 프로세스는 대기 상태
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C detected. Terminating all processes...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
    finally:
        print("[Main] Cleaning up shared memory.")
        shm.close()
        shm.unlink()
        sys.exit(0)
