import socket
import msg_pb2
from multiprocessing import Process
import os


class MultiClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
    def send_request(self):
        # 소켓 연결 설정
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((self.host, self.port))
            print('connected')

            while True:

                request_data = msg_pb2.Person()
                request_data.id = os.getpid()

                serialized_request = request_data.SerializeToString()
                print(f'request_data: {request_data}')
                print(f'serialized_request: {serialized_request}')
                msg_len = len(serialized_request).to_bytes(4, 'big')
                client_socket.sendall(msg_len+serialized_request)

                try:
                    size_bytes = client_socket.recv(4)
                    if not size_bytes:
                        break
                    resp_len = int.from_bytes(size_bytes, 'big')
                    resp_data = b''
                    while len(resp_data) < resp_len:
                        packet = client_socket.recv(resp_len - len(resp_data))
                        if not packet:
                            break
                        resp_data += packet

                    request_data.ParseFromString(resp_data)
                    print(f"Received response from server: {request_data}")

                except Exception as e:
                    print(f"Error receiving data: {e}")
                    break

# 테스트 실행 함수
def run_test(port):
    print("parent pid : %s, pid : %s" % (os.getppid(), os.getpid()))
    # 클라이언트 요청 테스트 실행
    client = MultiClient('localhost', port)
    client.send_request()

# 테스트 실행
if __name__ == "__main__":
        procs = []
        ports = [12345, 12346, 12347]
        for p in ports:
            proc = Process(target=run_test, args=(p,))
            procs.append(proc)
            proc.start()
        
        # 모든 프로세스의 종료 대기
        for proc in procs:
            proc.join()