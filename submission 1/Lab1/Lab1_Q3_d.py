# Lab 1, Question 3d - keep adding integers until the sum reaches 10,000
num = 0
sum=0
while sum<10000: #runs until the sum is above 10,000
    num+=1 # the number of iterations
    sum += num
print(sum)
print(num)
