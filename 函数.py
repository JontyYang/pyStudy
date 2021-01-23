# 函数的返回值是tuple
from functools import reduce


def my_abd(x):
    if not isinstance(x, (int, float)):
        raise TypeError('bad oprand error')
    if x >= 0:
        return x
    else:
        return -x

# Python函数在定义的时候，默认参数L的值就被计算出来了，
# 即[]，因为默认参数L也是一个变量，它指向对象[]，每次
# 调用该函数，如果改变了L的内容，则下次调用时，默认参数的
# 内容就变了，不再是函数定义时的[]了。
# 因此默认参数必须指向不变对象

# 可变参数(函数内部为tuple)[0或任意个]
def calc(*numbers):
    sum = 0
    for n in numbers:
        sum += n
    return sum
nums = [1, 2, 3]
print(calc(*nums))

# 关键字参数
# 可变参数允许你传入0个或任意个参数，这些可变参数在函数调用时自动组
# 装为一个tuple。而关键字参数允许你传入0个或任意个含参数名的参数，这
# 些关键字参数在函数内部自动组装为一个dict
def person(name, age, **kw):
    print('name:', name, 'age:', age, 'other:', kw)

person('jonty', 18, city='wuwei')
extra = {
    'city': 'wuwei',
    'job': 'engineer',
    'address': '123'
}
person('12', 23, **extra)

# 关键字参数**kw不同，命名关键字参数需要一个特殊分隔符*，
# *后面的参数被视为命名关键字参数。
def person(name, age, *, city, job):
    print(name, age, city, job)
person('Jack', 24, city='Beijing', job='Engineer')
# 如果函数定义中已经有了一个可变参数，后面跟着的命名关键字参数就
# 不再需要一个特殊分隔符*了：
def person(name, age, *args, city, job):
    print(name, age, args, city, job)

# 参数定义的顺序必须是：必选参数、默认参数、可变参数、命名关键字参数和关键字参数
# 对于任意函数，都可以通过类似func(*args, **kw)的形式调用它，无论它的参数是如何定义的。


# 高阶函数
# map/reduce
# map()函数接收两个参数，一个是函数，一个是Iterable，map将传入的函数依次
# 作用到序列的每个元素，并把结果作为新的Iterator返回。
def f(x):
    return x * x
r = map(f, [1, 2, 3, 4, 5])
print(list(r))
print(list(map(str, [1, 2, 3, 4, 5, 6, 7, 8, 9])))

# reduce把一个函数作用在一个序列[x1, x2, x3, ...]上，这个函数必须接收两个参数，
# reduce把结果继续和序列的下一个元素做累积计算
def add(x, y):
    return x + y
print(reduce(add, [1, 3, 5, 7, 9]))

# filter()把传入的函数依次作用于每个元素，然后根据返回值是True还是False决定保留还是丢弃该元素。
def is_odd(n):
    return n % 2 == 1
print(list(filter(is_odd, [1, 2, 4, 5, 6, 9, 10, 15])))