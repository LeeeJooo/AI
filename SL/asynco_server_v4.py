import socket
import msg_pb2
from multiprocessing import Process, Queue, Lock, Condition
from queue import Empty
import struct
import sys
import os
import time

class SharedMemory:
    def __init__(self):
        self.queue = Queue()
        self.data_list = []
        self.data_size = 0
        self.done = False
        self._lock = Lock()
        self._condition = Condition(self._lock)

    def add_data(self, data):
        with self._lock:
            self.queue.put(data)
            self.data_list.append(data)

            self._condition.notify()

    def check_queue(self):
        print(f"[p-{os.getpid()}] check_queue 시작")

        while not self.done:
            try:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self.queue.qsize() > self.data_size,
                        timeout = 5
                    )
                    self.print_data()
                    self.data_size = self.queue.qsize()

            except Exception as e:
                print(f"[p-{os.getpid()}] shared memory make batch 오류 발생: {e}")
                self.done=True
                time.sleep(1)
    
    def print_data(self):
        for i, d in enumerate(self.data_list):
            print(f'{i}: {d}')

def write_data_to_shared_memory(_shared_memory, data):
    _shared_memory.add_data(data)

def server_process(port, _shared_memory):
    print(f'>> [p-{os.getpid()}] START SERVER PROCESS with port {port}')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(('localhost', port))
        server_socket.listen()
        print(f"[PORT {port}] Server listening...")

        while True:
            client_socket, (c_addr, c_port) = server_socket.accept()

            with client_socket:
                print(f">> [p-{os.getpid()}/PORT {port}] SUCCESS TO CONNECTION WITH {c_addr}/{c_port}... Server listening...")
                try:
                    size_bytes = client_socket.recv(4)
                    size = int.from_bytes(size_bytes, 'big')
                    data = client_socket.recv(size)

                    msg = msg_pb2.Person()
                    msg.ParseFromString(data)
                    print(f">> [p-{os.getpid()}/PORT {port}] Request Message: {msg}")

                    write_data_to_shared_memory(_shared_memory, msg.id)

                except Exception as e:
                    print(f"[{port}] Error: {e}")

def start_check_queue(_shared_memory):
    _shared_memory.check_queue()

if __name__ == '__main__':
    print(f'[p-{os.getpid()}] START MAIN PROCESS')
    _shared_memory = SharedMemory()

    # Manage Shared Memory
    shared_memory_process = Process(target=start_check_queue, args=(_shared_memory,))
    shared_memory_process.start()

    # Server Process
    ports = [12345, 12346, 12347]
    server_processes = []

    # 서버 프로세스들 생성
    for port in ports:
        p = Process(target=server_process, args=(port, _shared_memory))
        p.start()
        server_processes.append(p)
    try:
        # 메인 프로세스는 대기 상태
        for p in server_processes:
            p.join()
    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C detected. Terminating all server_processes...")
        for p in server_processes:
            p.terminate()
        for p in server_processes:
            p.join()
    finally:
        print("[Main] Cleaning up shared memory.")
        sys.exit(0)
