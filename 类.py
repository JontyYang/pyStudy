class Animal(object):
    def run(self):
        print("Animal is running!!!")

class Cat(Animal):
    def run(self):
        print("Cat is running!!!")

class Dog(Animal):
    def run(self):
        print("Dog is running!!!")

# 鸭子类型
class Turtle(object):
    def run(self):
        print("turtle is Running!!!")
def run_twice(Animal):
    Animal.run()

run_twice(Cat())
run_twice(Dog())
run_twice(Turtle())
