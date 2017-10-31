from random import random

for _ in range(100000):
    x, y = random(), random()
    if (x < 0.5 and y < 0.5) or (x >= 0.5 and y >= 0.5):
        print(x, y, 1)
    else:
        print(x, y, -1)

print('-')

while True:
    print(random(), random())
