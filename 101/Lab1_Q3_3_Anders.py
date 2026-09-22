x=0
results_0=0
for i in range(0, 25):
   x= 2*i+1
   if x%3==0: # use the modulus operator to check if x is divisible by 3
      pass #this means do nothing if x is divisible by 3
   else:
      results_0+=x**2
print(results_0)