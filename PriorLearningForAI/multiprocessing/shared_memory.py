from multiprocessing import Process, shared_memory, Lock
import time

def worker(shm_name, lock, pid):
    start = time.time()

    shm = shared_memory.SharedMemory(name=shm_name)
    buffer = shm.buf
    iterations = int.from_bytes(buffer[4:8], byteorder='little')
    counter = int.from_bytes(buffer[:4], byteorder='little')
    
    print(f'[start] PID: {pid}, counter: {counter}, iteration: {iterations}')

    for _ in range(iterations):
        lock.acquire()
        counter = int.from_bytes(buffer[:4], byteorder='little') + 1
        buffer[:4] = counter.to_bytes(4, byteorder='little')
        
        if (_+1)%100000 == 0:
            print(f'STEP: {_+1}, PID: {pid}, counter: {counter}')
        
        lock.release()


    end = time.time()
    print(f"[PID : {pid}] Process : {counter} ({end-start:.2f} sec)")

# scope : main vs thread
# debug console

if __name__ == "__main__":
    shm = shared_memory.SharedMemory(create=True, size=10)
    buffer = shm.buf
    lock = Lock()

    counter = 0
    iterations = 10000000
    
    counter_byte = counter.to_bytes(4, byteorder='little')
    iterations_byte = iterations.to_bytes(4, byteorder='little')
    buffer[:4] = counter_byte
    buffer[4:8] = iterations_byte

    print(f'buffer size : {len(buffer)}')
    print(f"buffer[0] : {int.from_bytes(buffer[:4], byteorder='little')}")
    print(f"buffer[1] : {int.from_bytes(buffer[4:8], byteorder='little')}")

    processes = []
    for i in range(2):
        p = Process(target=worker, args=(shm.name, lock, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    
    shm.close()

# [과제]
# 스레드 출력 왜 다른지 :
# 멀티 프로세스 : 하나의 변수 접근 가능?
# 싱글 vs 멀티 스레드 속도 차이 > context switch
#  > 규모
#  > ...
# s, m, l >> lock