from __future__ import division, print_function

from pandas import Series
import numpy as np
from numpy.linalg import norm

def Point(xy):
    x,y = xy
    return Series({'x':x, 'y':y})

class Constraint:
    def __init__(self, f):
        self.f = f

    def __call__(self, x):
        return self.f(x)

    def linearize(self, x0):
        """
        c0 = g(x0)
        dc0 = dg(x)/dx|x0
        c ~ c0 + dc0'*(x-x0) = (c0 - dc0'*x0) + dc0'*x

        c <= 0 --> dc0'*x <= dc0'*x0 - c0
        """

        c0, dc0 = self.f(x0)
        A = dc0
        b = dc0.T.dot(x0) - c0
        return A, b

    @classmethod
    def from_obstacle(cls, obstacle, step0, step1, slack):
        step0 = Point(step0)
        step1 = Point(step1)
        def constraint_fun(x):
            pa = Point(x[step0])
            pb = Point(x[step1])
            po = obstacle.pt
            t = x[slack]
            nv = len(x)

            if all(pb == pa):
                n = pa - po
                distance = norm(n)
                alpha = (0.5, 0.5)
            else:
                u = pb - pa
                u = u / norm(u)
                a = np.array([-u[1], u[0]])
                b = a.dot(pa)

                g = a.dot(po) - b

                delta1 = u.dot(po) - u.dot(pa)
                delta2 = u.dot(po) - u.dot(pb)

                if np.sign(delta1) == np.sign(delta2):
                    if np.sign(delta1) == 1:
                        alpha = [0,1]
                        n = pb - po
                        distance = norm(n)
                    else:
                        alpha = [1,0]
                        n = pa - po
                        distance = norm(n)
                else:
                    if g >= 0:
                        n = -a
                    else:
                        n = a
                    distance = abs(g)
                    alpha = [abs(delta2) / (abs(delta1) + abs(delta2)), abs(delta1) / (abs(delta1) + abs(delta2))]
            c = obstacle.size - distance - t
            J_pa = np.zeros((2,nv))
            J_pa[0,step0['x']] = 1
            J_pa[1,step0['y']] = 1
            J_pb = np.zeros((2,nv))
            J_pb[0,step1['x']] = 1
            J_pb[1,step1['y']] = 1
            dc = -(alpha[0] * n.T.dot(J_pa) + alpha[1] * n.T.dot(J_pb))
            dc[slack] = -1
            return c, dc
        return cls(constraint_fun)