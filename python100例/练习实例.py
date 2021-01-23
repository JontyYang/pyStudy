# 题目：有四个数字：1、2、3、4，能组成多少个互不相同且无重复数字的三位数？各是多少？
# import datetime
# time1 = datetime.datetime.now()
# list = range(1, 5)
# count = 0
# # 64次循环
# for number1 in list:
#     for number2 in list:
#         for number3 in list:
#             if (number1 != number2) and (number1 != number3) and (number2 != number3):
#                 print(number1, number2, number3)
# # 少16次循环
# for number1 in list:
#     for number2 in list:
#         if(number1 == number2):
#             continue
#         else:
#             for number3 in list:
#                 if(number2 != number3) and (number3 != number1):
#                     print(number1, number2, number3)

# 企业发放的奖金根据利润提成。
# 利润(I)低于或等于10万元时，奖金可提10%；
# 利润高于10万元，低于20万元时，低于10万元的部分按10%提成，高于10万元的部分，可提成7.5%；
# 20万到40万之间时，高于20万元的部分，可提成5%；
# 40万到60万之间时高于40万元的部分，可提成3%；
# 60万到100万之间时，高于60万元的部分，可提成1.5%，高于100万元时，超过100万元的部分按1%提成，从键盘输入当月利润I，求应发放奖金总数？
# from click._compat import raw_input
#
# i = int(raw_input('净利润:'))
# arr = [1000000, 600000, 400000, 200000, 100000, 0]
# rat = [0.01, 0.015, 0.03, 0.05, 0.075, 0.1]
# r = 0
# for idx in range(0, 6):
#     if i > arr[idx]:
#         r += (i-arr[idx])*rat[idx]
#         print((i-arr[idx])*rat[idx])
#         i = arr[idx]
# print(r)

# 输入某年某月某日，判断这一天是这一年的第几天？
# year = int(input('year:'))
# month = int(input('month:'))
# day = int(input('day:'))
# months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
# month -= 1
# i = 0
# sum = 0
# if(year % 400 == 0) or ((year % 4 == 0) and (year % 100 != 0)):
#     months[1] = 29
# while month:
#     sum += months[i]
#     i += 1
#     month -= 1
# print(sum + day)

# 输入三个整数x,y,z，请把这三个数由小到大输出。
# l = []
# for i in range(1, 4):
#     x = int(input("请输入:"))
#     l.append(x)
# l.sort()
# print(l)

# 题目：斐波那契数列。
# 1、1、2、3、5、8、13、21、34
# def fib(n):
#     if n == 1 or n == 2:
#         return 1
#     else:
#         return fib(n-1) + fib(n-2)
# print(fib(10))

# 题目：输出 9*9 乘法口诀表
# for i in range(1, 10):
#     for j in range(1, i+1):
#         print("%d*%d=%d" % (i, j, i*j), end=" ")
#     print("\n")
# import datetime
# import time
#
# print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
# time.sleep(1)
# print(datetime.datetime.now())


# 题目： 判断素数
# from math import sqrt
# # 判断素数标准：不能够被2-n的平方根整除
# for m in range(101, 201):
#     # 素数出现的标志
#     flag = 1
#     k = int(sqrt(m))
#     for j in range(2, k+1):
#         if m % j == 0:
#             flag = 0
#             break
#     if flag == 1:
#         print(m)

# 判断水仙花数
# for m in range(100, 1000):
#     x = m // 100
#     y = m % 100 // 10
#     z = m % 10
#     if pow(x, 3) + pow(y, 3) + pow(z, 3) == m:
#         print(m)

# 输入一行字符，分别统计出其中英文字母、空格、数字和其它字符的个数。
# string = str(input("请输入字符串："))
# letters = 0
# space = 0
# number = 0
# others = 0
# for s in string:
#     if s.isalpha():
#         letters += 1
#     elif s.isdigit():
#         number += 1
#     elif s.isspace():
#         space += 1
#     else:
#         others += 1
# print('字母：%d'%letters)
# print('数字%d'%number)
# print('空格%d'%space)
# print('其他%d'%others)

# 求s=a+aa+aaa+aaaa+aa...a的值，其中a是一个数字。
# 例如2+22+222+2222+22222(此时共有5个数相加)，几个数相加由键盘控制。
# from functools import reduce
#
# List = []
# m = int(input('m='))
# n = int(input('n='))
# sum = 0
# for x in range(n):
#     sum = sum * 10 + m
#     List.append(sum)
#     print(sum)
# print(reduce(lambda x, y : x + y, List))

# 题目：一个数如果恰好等于它的因子之和，这个数就称为"完数"。
# 例如6=1＋2＋3.编程找出1000以内的所有完数。
# for number in range(2, 1001):
#     L = []
#     sum = 0
#     for i in range(1, number):
#         if number % i == 0:
#             L.append(i)
#     for list in L:
#         sum += list
#     if sum == number:
#         print(number)
#         print(L)

# 题目：一球从100米高度自由落下，每次落地后反跳回原高度的一半；再落下，
# 求它在第10次落地时，共经过多少米？第10次反弹多高？
# sum = 0.0
# height = 100.0
# for i in range(10):
#     sum += height
#     height *= 0.5
#     sum += height
#     print('高度为%f'% height)
#     print('总路程为%f' % sum)

# l = [1, 2, 3, 4, 5]
# s1 = ','.join(str(n) for n in l)
# print(s1)

# 匿名函数
# MAXIMUM = lambda x,y :  (x > y) * x + (x < y) * y
# MINIMUM = lambda x,y :  (x > y) * y + (x < y) * x


# 从键盘输入一些字符，逐个把它们写到磁盘文件上，直到输入一个 # 为止
# with open('./file1.txt', 'w', errors='ignore') as fp:
#     ch = input('string')
#     for char in ch:
#         if char != '#':
#             fp.write(char)
#         else:
#             break