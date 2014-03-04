from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import gurobipy as grb
from numpy.linalg import norm
from constraint import Constraint

class Trajopt:
    mu_0 = 1
    s_0 = 1
    iter_limit = 50
    improvement_threshold_c = 0.8
    tau_plus = 1.5
    tau_minus = 0.5
    penalty_multiplier_k = 2
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
        self.convex_iter_callback = lambda x: None

        self.generate_seed(p0, pf)
        self.build_constraints(obstacles)

    def generate_seed(self, p0, pf):
        nsteps = self.steps_i.shape[1]
        nslacks = self.slacks_i.shape[0]
        self.x0 = np.zeros(self.nv)
        self.lb = np.zeros_like(self.x0)
        self.ub = np.zeros_like(self.x0)

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
            self.x0[self.steps_i[:,i]] = steps0[:,i]
            self.lb[self.steps_i[:,i]] = steps_lb[:,i]
            self.ub[self.steps_i[:,i]] = steps_ub[:,i]

        t0 = np.zeros(nslacks)
        t_lb = np.zeros_like(t0)
        t_ub = np.inf * np.ones_like(t0)
        for i, ti in enumerate(t0):
            self.x0[self.slacks_i[i]] = ti
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
        """
        Compute the (quadratic) base cost.
        This could be vectorized for speed, but doing it in a very basic
        way like this means that we can use exactly the same code to
        compute the numerical cost and to build the gurobi cost expression.
        """
        obj = 0
        for i in range(self.steps_i.shape[1]-1):
            for j in [0, 1]:
                obj += (x[self.steps_i[j,i]] - x[self.steps_i[j,i+1]])*(x[self.steps_i[j,i]] - x[self.steps_i[j,i+1]])
        return obj

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

        obj = self.base_cost(grb_vars)
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
                        expr += ai[k] * var
                    # import pdb; pdb.set_trace()
                    # print("adding constraint, ai:", ai, "bi:", bi)
                    m.addConstr(expr <= bi, "c{:d}".format(i))

        m.setObjective(obj)
        m.update()

        return m, grb_vars

    def trust_region_iteration(self, x, cost, model, grb_vars):
        # print("new trust region iteration")
        for step in self.steps_i.T:
            for idx in step:
                grb_vars[idx].lb = float(max(self.lb[idx], x[idx] - self.trust_s))
                grb_vars[idx].ub = float(min(self.ub[idx], x[idx] + self.trust_s))
        model.optimize()
        xstar = np.array([v.x for v in grb_vars])
        model_improve = cost - model.objVal
        true_cost = self.nonconvex_cost(xstar)
        true_improve = cost - true_cost
        trustworthy = true_improve / model_improve > self.improvement_threshold_c
        return xstar, true_cost, trustworthy

    def convexify_iteration(self, x, cost):
        # print("new convexify iteration")
        model, grb_vars = self.convexify(x)
        while True:
            xstar, cstar, trustworthy = self.trust_region_iteration(x, cost, model, grb_vars)
            if trustworthy:
                xdiff = max(np.abs(x - xstar).flat)
                self.trust_s *= self.tau_plus
                break
            else:
                self.trust_s *= self.tau_minus
                if self.trust_s < self.xtol:
                    xdiff = max(np.abs(x - xstar).flat)
                    break
        self.convex_iter_callback(xstar)
        return xstar, cstar, xdiff

    def penalty_iteration(self, x):
        # print("new penalty iteration")
        cost = self.nonconvex_cost(x)
        while True:
            xstar, cstar, xdiff = self.convexify_iteration(x, cost)
            if cstar < self.ftol or xdiff < self.xtol:
                break
            x = xstar
            cost = cstar
        return xstar

    def run_scp(self, x, convex_iter_callback = lambda x: None):
        self.pentalty_mu = self.mu_0
        self.trust_s = self.s_0
        self.convex_iter_callback = convex_iter_callback
        iters = 0
        while True:
            x = self.penalty_iteration(x)
            iters += 1
            y = x.copy()
            y[self.slacks_i] = 0
            if all(c(y)[0] < self.ctol for c in self.obstacle_constraints):
                print("Finished")
                ok = True
                break
            if iters > self.iter_limit:
                print("Failed: iter limit reached")
                ok = False
                break

            self.pentalty_mu *= self.penalty_multiplier_k
        return x, ok

    def draw(self, x, ax):
        plt.cla()
        path = ax.plot(x[self.steps_i[0,:]], x[self.steps_i[1,:]], 'k-o')
        for c in self.obstacle_constraints:
            c.draw(ax)
        return path

