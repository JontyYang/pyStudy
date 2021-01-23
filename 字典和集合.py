dict = {
    1: 'Java',
    2: 'Python',
    3: 'C',
    4: 'C++',
    5: 'JS'
}
 # 判断字典中是否含有某键值
# print(dict.get(7), -1)
# print(1 in dict)
# print(dict.pop(5))
# 遍历value
# for value in dict.values():
#     print(value)
# 遍历dict
# for key, value in dict.items():
#     print(key, value)
# enumerate将list或其它类型转换为索引-元素对
# for (index, key) in enumerate(dict):
#     print(index, key, end=' ')
#     print(dict[value])


# 列表生成式(for可以循环多个变量，因此列表生成式可以使用多个变量生成）
# if在for之后没有else，作为删选条件
# list = [x * x for x in range(0, 11) if x % 2 == 0]
# print(list)
# print([m + n for m in 'ABC' for n in 'XYZ'])
# if在for之前，作为表达式，有else
# print([x if x % 2 == 0 else -x for x in range(1, 11)])

# 在循环的过程中不断推算出后续的元素,这样就不必创建完整的list，从而节省大量的空间。
# 将列表生成式[]变为()为列表生成器
# 将函数中添加yield为生成器
def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n = n + 1
    return 'done'
for x in fib(8):
    print(x)