import sys
sys.setrecursionlimit(100000)
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return(fibonacci(n-1) + fibonacci(n-2))


print(fibonacci(1000))