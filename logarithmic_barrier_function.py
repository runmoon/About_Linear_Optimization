"""
Interior point algorithm based on affine scaling with logarithmic barrier function
"""

from numpy import *

# initial
A = [[1, -1, 1, 0], [0, 1, 0, 1]]
b = [15, 15]
c = [-2, 1, 0, 0]
x0 = [10, 2, 7, 13]
A = mat(A)
b = mat(b).T
c = mat(c).T
x0 = mat(x0).T
zero_p = 0.000001  # test b-Ax=0, primary feasibility
zero_c = 0.000001  # test r=c-Ap=0, dual feasibility
zero_d = 0.000001  # test xr=xc-bp=0, compensating relaxation condition
(m, n) = shape(A)
a = 0.99  # use for minimum ratio test
coe_log = 0.01

iteration = 0
iteration_limit = 20
while iteration < iteration_limit:
    iteration = iteration+1
    print('\niteration = ', iteration)
    print('xT = ', x0.T)
    print("cx = ", c.T * x0)
    # dual vector estimate
    X = eye(n)  # affine scale transformation matrix
    for i in range(n):
        X[i][i] = x0[i, 0]
    X = mat(X)
    temp1_AXX = A*X*X
    p = (temp1_AXX*A.T).I * temp1_AXX * c
    # print('p = ', p)
    # reduced cost vector
    r = c - A.T * p
    # dual_gap
    dual_gap = x0.T * r
    # dual_gap = c.T*x0 - b.T*p
    r_minus = []
    for i in range(n):
        if r[i, 0] < 0:
            if abs(r[i, 0])/(abs(c[i,0]) + 1) > zero_d:
                r_minus.append(i)
    # print('r_minus = ', r_minus)
    # check optimization
    if r_minus and (abs(dual_gap) < zero_c):
        print('dual_gap =', dual_gap)
        break

    # transfer direction
    dy = -X * r
    # print('dyT = ', dy.T)
    if dy.T * dy < 0.001:  # # regard as converged, has reached optimization solution
        break
    # step size, using minimum ratio test
    ak = 0
    a_test = []
    for i in range(n):
        if i not in r_minus:  # when dy(i) is minus
            a_test.append(a/-(dy[i, 0]))
    ak = min(a_test)
    print("ak = ", ak)
    # a new solution
    dx = X * dy
    print('dxT = ', dx.T)
    derivative_log = []
    for i in range(n):
        derivative_log.append(1/x0[i, 0])
    derivative_log = mat(derivative_log)
    center_force_direction = (X - X*A.T*(temp1_AXX*A.T).I * temp1_AXX) * derivative_log.T
    x = x0 + ak * (dx + coe_log * center_force_direction)
    if (x-x0).T*(x-x0) < 0.1:  # regard as converged
        break
    x0 = x
print('x = ', x)
print("cx = ", c.T * x0)