import asyncio
import socket
import msg_pb2
from multiprocessing import Process

HOST = 'localhost'
PORT = 12345

async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    addr = writer.get_extra_info('peername')
    print(f"Connected by {addr}")

    try:
        # 메시지 크기를 먼저 받음 (예: 4바이트로)
        size_bytes = await reader.readexactly(4)
        size = int.from_bytes(size_bytes, byteorder='big')
        data = await reader.readexactly(4)
        print(f'size_byte: {data}')
        size = int.from_bytes(size_bytes, byteorder='big')
        print(f'size2: {size}')
        # data = await reader.readexactly(size)
        # print(f'data: {data}')
        msg = msg_pb2.Person()
        msg.ParseFromString(size_bytes)
        print(f'msg: {msg}')

        print(f"Received message from {addr}: {msg}")

    except asyncio.IncompleteReadError:
        print(f"Connection lost from {addr}")
    finally:
        writer.close()
        await writer.wait_closed()
        print(f"Disconnected from {addr}")

async def run_server():
    server = await asyncio.start_server(handle_client, HOST, PORT)
    print(f'Server started on {HOST}:{PORT}')
    async with server:
        await server.serve_forever()

if __name__ == '__main__':
    procs = []
    for i in range(3):
        proc = Process(target=run_server)
        procs.append(proc)
        proc.start()
    
    # 모든 프로세스의 종료 대기
    for proc in procs:
        proc.join()
    asyncio.run(run_server())
