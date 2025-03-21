from multiprocessing import Process, Manager, Lock
import time

def worker(lock, counter, iterations, pid):
    start = time.time()

    print(f'[start] PID: {pid}, counter: {counter.value}, iteration: {iterations.value}')
    for _ in range(iterations.value):
        lock.acquire()
        counter.value += 1
        lock.release()

        if (_+1)%100000 == 0:
            print(f'STEP: {_+1}, PID: {pid}, counter: {counter.value}')

    end = time.time()
    print(f"[PID : {pid}] Process : {counter} ({end-start:.2f} sec)")

# scope : main vs thread
# debug console

if __name__ == "__main__":
    with Manager() as manager:
        counter = manager.Value(int, 0)
        iterations = manager.Value(int, 10000000)
        lock = Lock()

        print(f"counter : {counter.value}")
        print(f"iterations : {iterations.value}")

        processes = []
        for i in range(2):
            p = Process(target=worker, args=(lock, counter, iterations, i))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
        

# [과제]
# 스레드 출력 왜 다른지 :
# 멀티 프로세스 : 하나의 변수 접근 가능?
# 싱글 vs 멀티 스레드 속도 차이 > context switch
#  > 규모
#  > ...
# s, m, l >> lock