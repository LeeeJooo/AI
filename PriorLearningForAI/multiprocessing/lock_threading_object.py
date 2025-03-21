import time
import threading

counter = 0
iterations = 10000000

# 동기화 딜레이
def increment(tid):
    global counter, iterations
    start = time.time()
    
    for _ in range(iterations):
        lock_.acquire()
        
        temp = counter
        temp += 1
        counter = temp

        lock_.release()

    end = time.time()

    print(f'[TID : {tid}] {end-start:.2f} sec')

# scope : main vs thread
# debug console

if __name__ == "__main__":     
    print('LOCK THREADING OBJECT')
    lock_ = threading.Lock()

    threads = []
    for i in range(2):
        t = threading.Thread(target=increment, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"ANSWER: {2*iterations}, THREAD OUTPUT: {counter}")