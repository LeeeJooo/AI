import socket

# TCP 소켓 생성
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 서버의 주소와 포트로 연결
client_socket.connect(('localhost', 12345))

# 서버에 데이터 전송
client_socket.sendall(b'Hello, server!')

# 서버로부터 에코 받은 데이터 출력
data = client_socket.recv(1024)
print(f"Received echo: {data.decode()}")

# 소켓 닫기
client_socket.close()
