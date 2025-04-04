import socket
import msg_pb2

# TCP 소켓 생성
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 주소 재사용 옵션 설정
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# 소켓을 localhost의 12345 포트에 바인드
server_socket.bind(('localhost', 12345))

# 연결 요청 대기
server_socket.listen()

print("Server waiting for connections...")
client_socket, addr = server_socket.accept()
print(f"Connected by {addr}")

# 클라이언트로부터 데이터를 받고, 그대로 에코로 보냄
data = client_socket.recv(1024)
msg = msg_pb2.Person()
msg.ParseFromString(data)
print(f"Received data: {data}")
# client_socket.sendall(data)

# 소켓 닫기
client_socket.close()
server_socket.close()
