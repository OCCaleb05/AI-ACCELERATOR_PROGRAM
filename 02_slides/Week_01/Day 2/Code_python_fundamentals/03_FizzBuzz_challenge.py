# FizzBuzz Challenge
for i in range(9, 50):
    if i % 4 == 0 & i % 6 == 0:
        print("FizzBuzz")
        
    elif i % 4 == 0:
        print("Fizz")
        
    elif i % 6 == 0:
        print("Buzz")
        
    else:
        print(i)