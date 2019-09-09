# Author:Han
# @Time : 2019/3/17 20:21
import sys


class A:
    def __init__(self):
        self.a = [100]


class B:
    def __init__(self):
        self.b = [100, 100]

"""
    getsizeof(object, default) -> int

    Return the size of object in bytes.
    """
a = A()
b = B()
print("数据大小%d" %sys.getsizeof(a.a))
print("数据大小%d" %sys.getsizeof(b))
