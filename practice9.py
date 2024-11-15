import torch  # 导入pytorch库


def fX():
    # 创建变量，并赋初值，为0-3的向量
    x = torch.range(0.0, 3.0)
    print("x is ",x)  # 查看x 的值

    x.requires_grad_(True)
    print("x.grad is: ", x.grad)

    # 计算y,y=2xTx
    y = 2 * torch.dot(x, x)
    print("firstly y:", y)
    # 计算y关于x每个分量的梯度
    y.backward()
    print("x.grad is ", x.grad)
    # 函数 𝑦=2𝐱⊤𝐱关于 𝐱的梯度应为 4𝐱。
    # 让我们快速验证这个梯度是否计算正确。
    print(x.grad == 4 * x)

    # 默认情况下pytorch会累积梯度，因此需要清楚之前的值
    x.grad.zero_()
    y = x.sum()
    print("later of the x.sum() y:", y)
    # 计算y关于变量的梯度
    y.backward()
    print("x.grad is ", x.grad)

    print("----------------------------------------")
    """非标量变量的反向传播"""
    # 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
    # 本例只想求偏导数的和，所以传递一个1的梯度是合适的
    # 清除之前的梯度值，避免多次求梯度时导致爆内存
    x.grad.zero_()
    y = x * x
    # 等价于y.backward(torch.ones(len(x)))
    y = y.sum()
    # 查看y的数据情况
    print("y is ", y)
    # 调用反向传播函数来自动计算y关于x每个分量的梯度
    y.backward()
    # 打印梯度
    print(x.grad)

    """分离计算"""
    """有时，我们希望[将某些计算移动到记录的计算图之外]。 
    例如，假设y是作为x的函数计算的，而z则是作为y和x的函数计算的。
    想象一下，我们想计算z关于x的梯度，但由于某种原因，
    希望将y视为一个常数， 并且只考虑到x在y被计算后发挥的作用。
    这里可以分离y来返回一个新变量u，该变量与y具有相同的值，
    但丢弃计算图中如何计算y的任何信息。 换句话说，
    梯度不会向后流经u到x。 因此，
    下面的反向传播函数计算z=u*x关于x的偏导数，
    同时将u作为常数处理， 而不是z=x*x*x关于x的偏导数。"""

    print("---------------------------------------------")
    # 清除之前的梯度
    x.grad.zero_()
    y = x * x
    # 创建y的一个副本u
    u = y.detach()
    z = u * x
    z.sum().backward()
    print("x.grad == u:", x.grad == u)
    # 由于记录了y的计算结果，我们可以随后在y上调用反向传播， 得到y=x*x关于的x的导数，即2*x
    x.grad.zero_()
    y.sum().backward()
    print("x.grad == 2 * x:", x.grad == 2 * x)

    """Python控制流的梯度计算"""
    print("------------------------------------")


fX()


# 使用自动微分的一个好处是：
# [即使构建函数的计算图需要通过Python控制流
# （例如，条件、循环或任意函数调用），
# 我们仍然可以计算得到的变量的梯度]。
# 在下面的代码中，
# while循环的迭代次数和if语句的结果都取决于输入a的值。

def f(a):
    b = a * 2
    # 打印b的数据类型
    print(b.type())
    # 打印b的值
    print(b)
    # 打印b的范数（即向量的模）
    print(b.norm())
    #  # 当b的范数小于1000时，循环
    while b.norm() < 1000:
        b = b * 2
        # 如果b的所有元素之和大于0
        if b.sum() > 0:
            c = b
        else:
            c = 100 * b
        return c


#  创建一个随机初始化的张量随机数a，维数为0，要求梯度，并初始化为0
a = torch.randn(size=(), requires_grad=True)
# 打印a 的情况
print("a:", a)
d = f(a)
# 调用backward方法，计算d关于a的梯度
d.backward()
print(a.grad == d / a)
