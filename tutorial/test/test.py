




a=[[0]*10]*10


def fib(n):
    if n == 1:
        return 1
    if n == 0:
        return 0
    return fib(n-1)+fib(n-2)

def testlist(a,b):
    a[0][b]=1

def testin(a):
    b=[[0]*10]*10
    testlist(b,a)
    print(b[0])

def testunzip(a,b):
    print(a,b)

def testbas(a):
    a=1
    print(a)

def plus(a,b):
    return a+b

    


if __name__ == '__main__':
    a=1
    b=2
    c,d=a,b
    print(c,d)
