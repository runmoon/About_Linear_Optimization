"""
this program can solve standard form of LP by using modified simplex algorithm;
in this program initial permissible solution is obtained by two-stage method
and Blend rule is used to choose the base vectors to out or in base.
what you need to input  is c, A, and b.

Standard form of LP:
min: cx'
s.t.:Ax'=b;
     x>=0
note:
x is a n×1 variables vector, and n is the number of the variables；
c is a n×1 coefficient vector(constant vector) of Objective function;
A and b is parameters of constraint equations; A is a m×n matrix, b is a m×1 vector,
m is the number of constraint equations.
"""

from numpy import *
import MSM_function

# Input Initial parameters
print("Standard form of LP is:\n min: cx'\n s.t.:Ax'=b;\n      x>=0\nPlease inter parameter A, b and c respectively:")
c = input('1.Input coefficient vector of Objective function c(separated each number by space):\n')
c = array([float(i) for i in c.split()])    # convert a string to a number vector
n = len(c)
m = int(input('2.Input the number of constraint equations m:\n'))
A = [[0]*n]*m  # initial A
print('3.Input the coefficient matrix of constraint equations A(separated each row by line break):')
for i in range(m):
    A[i] = [float(j) for j in input().split()] # input each row of A
A = array(A)
b = input('Input parameter vector of constraint equations b:\n')
b = array([float(i) for i in b.split()])  # convert a string to a number vector

print('A=\n', A)
print('b=\n', b)
print('c=\n', c)

# test1, redundant constraint equation
# A = [[1,0,-2,1], [1,1,1,0], [2,0,-4,2]]
# b = [1,4,2]
# c = [1, 1, 2, 5]
# n = 4
# m = 3

# test2, degenerate solution
# A = [[1, -2, 4], [4, -9, -3]]
# b = [4,16]
# c = [1, 2, 3]
# n = 3
# m = 2

# A = array(A)
# b = array(b)
# c = array(c)

Ib = array(range(m))
B_inv = array(eye(m))
solution = 1

# to check if A is standard LP
if ~(A[:, 0:m] == eye(m)).all():  # if A is not standard LP
    # step 1
    print('\n----step 1----', )
    A, b, Ib, B_inv, solution = msd_function.step1(A, b)

B = A[:, Ib]
(m, n) = shape(A)

if solution == 0:
    print('LP is unsolvable')
elif solution == 10:
    print('Reached the upper limit of the number of iteration.')
else:
    print('\n----step 2----', )
    # step2
    Ib, B_inv, solution = msd_function.modified_simplex_method(A, b, c, Ib, B_inv)

    print('\n')
    if solution == 2:
        print('The solution of this LP is unbounded.')
    elif solution == 10:
        print('Reached the upper limit of the number of iteration.')
    else:
        xb = dot(B_inv, b)
        x = array([0] * n)
        for i in range(m):
            x[Ib[i]] = xb[i]
        z = dot(c, x)

        print('Reached the optimal basic permissible solution:')
        print('  x = ', x)
        print('The optimal value of objective function is: \n  z = ', z)

