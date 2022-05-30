class A:
    @classmethod
    def foo(cls, a):
        print(cls)
        print(a)


A.foo(10)
