from numpy import *


def modified_simplex_method(A, b, c, Ib, B_inv):
    (m, n) = shape(A)
    B = A[:, Ib]
    cb = c[Ib]
    xb = dot(B_inv, b)
    x = array([0]*n)
    for i in range(m):
        x[Ib[i]] = xb[i]
    w = dot(cb, B_inv)  # simplex multiplier

    # 求r,既约费用
    q = -1
    for j in range(n):  # Bland 入基规则
        if j not in Ib:
            if c[j] - dot(w, A[:, j]) < 0:  # reduced cost vector about base B or discriminant vector about base B
                q = j  # 选定列q
                break
    if q != -1:
        print('入基列q=', q)

    solution = 1
    iteration = 0
    max_of_iteration = 1000  # the upper limit of the time of iteration
    while q > -1:  # 继续迭代条件：q>=0 ---> q>-1,
        # 检查迭代次数
        if iteration > max_of_iteration:
            break
            solution = 10
        iteration = iteration + 1

        d = - dot(B_inv, A[:, q])
        #  判断是否无下界
        dt = 0  # 记录d中小于0的元素的个数
        d_index = []  # 记录d中小于0的元素在d中的序号
        for i in range(m):
            if d[i] < 0:
                dt = dt + 1
                d_index.append(i)
        if len(d_index) == 0:
            solution = 2
            break  # 问题无下界，停止

        # 求jp，确定出基向量是A中的哪一列
        p = d_index[0]
        if dt > 1:  # Bland 出基规则
            for i in range(dt - 1):  # 在满足小于0的d中的元素中找
                if (-xb[d_index[i + 1]] / d[d_index[i + 1]]) < (-xb[d_index[i]] / d[d_index[i]]):
                    p = d_index[i + 1]  # B中的哪一列
                    # - x[I[m]] / d[i]
        jp = Ib[p]  # A中的哪一列
        print('出基列jp=', Ib[p])  # 出基列

        #  换基
        #  换基 法一
        x[q] = -x[jp] / d[p]  # 入基
        for i in range(m):
            x[Ib[i]] = x[Ib[i]] + x[q] * d[i]  # 出基
        ep = array([0] * m)
        ep[p] = 1
        B = B + array(dot(mat(A[:, q] - A[:, jp]).T, mat(ep)))
        B_inv = array(mat(B).I)
        Ib[p] = q

        #  换基 法二 （unfinished...）
        # M = mat([0]*m)*m
        # for i in range(m):
        #     M[i, p] = - d[i]/d[p]
        # xb = M * xb
        # B_inv = M * B_inv

        print('iteration = ', iteration)
        # 求r,既约费用
        cb = c[Ib]
        w = dot(cb.T, B_inv)  # simplex multiplier
        # w = cb.T * B_inv
        q = -1
        for j in range(n):  # Bland 入基规则
            if c[j] - dot(w, A[:, j]) < 0:
                q = j  # 选定列
                print('\n入基列q=', q)
                break
    return Ib, B_inv, solution


def step1(A, b):

    (m, n) = shape(A)
    m0 = m  # record initial n,m
    n0 = n
    n = n + m
    Ib = array(range(m))
    In = array([0] * (n-m))
    c_art = array([1]*m + [0]*(n-m))  # c_art is a artificial vector of artificial objective function
    A = hstack((array(eye(m)), A))
    B_inv = array(eye(m))

    Ib, B_inv, solution = modified_simplex_method(A, b, c_art, Ib, B_inv)

    B = A[:, Ib]
    N = array([[0]*(n - m)] * m)

    artI = []  # record sequence of the artificial vectors in base
    oriI = []  # record sequence of the original vectors in base
    solution = 1
    xb = dot(B_inv, b)
    x = array([0]*n)
    for i in range(m):
        x[Ib[i]] = xb[i]
    x = array(x)
    if sum(x * c_art) == 0:  # 目标函数的最优值为零
        for i in range(m):
            if Ib[i] < m:
                artI.append(i)  # artI记录了Ib中小于m的元素在Ib中的序号
            else:
                oriI.append(i)
    else:  # 目标函数的最优值不为零，原LP无解
        solution = 0

    for k in artI:  # 人工向量未全从基中退出,artI is not empty
        Ak_zero = []
        # 确定N
        j = 0
        for i in range(n):
            if i not in Ib:
                N[:, j] = A[:, i]
                In[j] = i
                j = j + 1
        Ak_row = dot(B_inv[Ib[k], :], N)  # 应该是线性变换后的Ak
        for i in range(n - m):
            if In[i] > (m - 1):
                if Ak_row[i]:  # record when A[element_artI,j]！=0
                    Ak_zero.append(In[i])
                    break

        if Ak_zero:  # 中至少一个不为零，换基
            inB = Ak_zero[0]  # 入基
            outB = Ib[k]  # 出基 Ib(k)
            ep = array([0] * m)
            ep[k] = 1
            B = B + array(dot(mat(A[:, inB] - A[:, outB]).T, mat(ep)))
            B_inv = array(mat(B).I)
            Ib[k] = inB
        else:  # 中全部为零,Ib(k)行多余了
            A = delete(A, Ib[k], axis=0)  # delete row A[Ib(k),:]
            b = delete(b, Ib[k], axis=0)
            Ib = delete(Ib, k, axis=0)

    Ib = Ib - m
    A = A[:, m:n]
    B = A[:, Ib]
    B_inv = array(mat(B).I)

    if artI:
        t = len(artI) - (m0 - len(Ib))
        if t:
            print('We will get a degenerate solution with %d base variables is zero.' % t)
        if m0 > len(Ib):
            print('There are %d equations in the constraint equations of the LP is redundant.' % (m0-len(Ib)))
    return A, b, Ib, B_inv, solution