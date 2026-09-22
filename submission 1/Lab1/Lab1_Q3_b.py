# Lab 1, Question 3b - sum the squares of all odd integers from 1 to 50
results_0=0
x=0
for i in range(0, 25): #there is 25 odd numbers between 0 and 50
    x= 2*i+1 #formula for odd numbers
    results_0+=x**2 #sum of squares of odd numbers
print(results_0)
