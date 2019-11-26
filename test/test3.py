
try: 
    datas = [] 
      
    while True: 
        datas.append(int(input()))   
except: 
    datas = datas

def counter(n):
    count = 0
    while n >= 1:
        if n == 1:
            n -= 1
            count += 1
        elif (n % 2) == 0:
            n = n/2
            count += 1
        else:
            n = (3 * n) + 1
            count += 1  

    return count

def nxtFibonacci(datas):
    D = datas[1:]
    outs = []
    for n in D:
        index = D.index(n) + 1
        N = counter(n)
        formatN = str(index) + " " + str(n) + " " + str(N)
        outs.append(formatN)
    
    for out in outs:
        print(out) 

nxtFibonacci(datas)