import time
import threading

# 힙
# counter = 0
# iterations = 10000000
# lock_ = False

# 동기화 딜레이
def increment(tid):
    # global lock_, counter, iterations
    start = time.time()

    i = 0
    while (i < iterations[0]):
        if lock_[0] :
            continue
        else :
            lock_[0]=True
            temp = counter[0]
            temp += 1
            counter[0] = temp
            i+=1
            lock_[0]=False

    end = time.time()
    print(f'[TID : {tid}] {end-start:.2f} sec')

# scope : main vs thread
# debug console

if __name__ == "__main__":
    print('LOCK BOOL')
    threads = []
    counter = [0]
    iterations = [10000000]
    lock_ = [False]

    for i in range(2):
        t = threading.Thread(target=increment, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"ANSWER: {2*iterations[0]}, THREAD OUTPUT: {counter}")
