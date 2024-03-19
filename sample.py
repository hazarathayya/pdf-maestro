import os

count = 0
for i in os.listdir('./data/Images'):
    count =count+1

print(count)