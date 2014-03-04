from __future__ import division, print_function
import numpy as np
import gurobipy as grb
from numpy.linalg import norm
from constraint import Constraint

class Trajopt:
    mu_0 = 1
    s_0 = 1
    improvement_threshold = 0.8
    tau_plus = 1.5
    tau_minus = 0.5
    k = 2
    ftol = 2e-4
    xtol = 0.01
    ctol = 0
    dim = 2
    d_check = 0.1

    def __init__(self, p0, pf, obstacles, num_knots):
        num_slacks = (num_knots - 1) * len(obstacles)
        num_step_vars = self.dim * num_knots
        self.nv = num_step_vars + num_slacks
        self.steps_i = np.array(range(num_step_vars)).reshape((-1,self.dim)).T
        self.slacks_i = np.array(range(num_step_vars, num_step_vars + num_slacks))

        self.generate_seed(p0, pf)
        self.pentalty_mu = self.mu_0
        self.build_constraints(obstacles)

    def generate_seed(self, p0, pf):
        nsteps = self.steps_i.shape[1]
        nslacks = self.slacks_i.shape[0]
        self.x = np.zeros(self.nv)
        self.lb = np.zeros_like(self.x)
        self.ub = np.zeros_like(self.x)

        px0 = np.linspace(p0[0], pf[0], nsteps)
        py0 = np.linspace(p0[1], pf[1], nsteps)
        steps0 = np.vstack((px0, py0))
        steps_lb = -np.inf * np.ones_like(steps0)
        steps_ub = np.inf * np.ones_like(steps0)
        steps_lb[:,0] = p0
        steps_ub[:,0] = p0
        steps_lb[:,-1] = pf
        steps_ub[:,-1] = pf
        for i in range(steps0.shape[1]):
            self.x[self.steps_i[:,i]] = steps0[:,i]
            self.lb[self.steps_i[:,i]] = steps_lb[:,i]
            self.ub[self.steps_i[:,i]] = steps_ub[:,i]

        t0 = np.zeros(nslacks)
        t_lb = np.zeros_like(t0)
        t_ub = np.inf * np.ones_like(t0)
        for i, ti in enumerate(t0):
            self.x[self.slacks_i[i]] = ti
            self.lb[self.slacks_i[i]] = t_lb[i]
            self.ub[self.slacks_i[i]] = t_ub[i]
        return self

    def build_constraints(self, obstacles):
        self.obstacle_constraints = []
        slack_num = 0
        for i in range(self.steps_i.shape[1]-1):
            for obs in obstacles:
                self.obstacle_constraints.append(
                     Constraint.from_obstacle(obs,
                                              self.steps_i[:,i],
                                              self.steps_i[:,i+1],
                                              self.slacks_i[slack_num]))
                slack_num += 1
        return self

    def base_cost(self, x):
        steps = x[self.steps_i]
        dx = np.diff(steps)
        dists = [norm(dx[:,i]) for i in range(dx.shape[1])]
        return sum(np.power(dists, 2))

    def nonconvex_cost(self, x):
        y = np.copy(x)
        y[self.slacks_i] = 0
        cost = self.base_cost(y)
        for con in self.obstacle_constraints:
            c, dc = con(y)
            c = max(c, 0)
            cost += c * self.pentalty_mu
        return cost

    def convexify(self, x):
        m = grb.Model('trajopt')
        grb_vars = []
        for i, _ in enumerate(x):
            grb_vars.append(m.addVar(lb=self.lb[i], ub=self.ub[i], name="x{:d}".format(i)))
        m.update()

        obj = 0
        for i in range(self.steps_i.shape[1]-1):
            for j in [0,1]:
                obj += (grb_vars[self.steps_i[j,i]] - grb_vars[self.steps_i[j,i+1]])*(grb_vars[self.steps_i[j,i]] - grb_vars[self.steps_i[j,i+1]])
        for i in range(self.slacks_i.size):
            obj += self.pentalty_mu * grb_vars[self.slacks_i[i]]
        for i, con in enumerate(self.obstacle_constraints):
            c, dc = con(x)
            if c > -self.d_check:
                A, b = con.linearize(x)
                A = A.reshape((-1, len(x)))
                b = b.reshape((A.shape[0],))
                for j in range(A.shape[0]):
                    ai = A[j,:]
                    bi = b[j]
                    expr = 0
                    for k, var in enumerate(grb_vars):
                        expr += var * ai[k]
                    # import pdb; pdb.set_trace()
                    print("adding constraint, ai:", ai, "bi:", bi)
                    m.addConstr(expr <= bi, "c{:d}".format(i))

        m.setObjective(obj)
        m.update()

        return m, grb_vars
