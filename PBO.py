from typing import Any
from problem.basic_problem import Basic_Problem
from optimizer import PSO, Particle
from optimizer_template import optimizer_template

class PBO_Env:
    """
    An environment with a problem and an optimizer.
    """
    def __init__(self,
                 problem: Basic_Problem,
                 optimizer: optimizer_template,
                 ):
        self.problem = problem
        self.optimizer = optimizer

    def reset(self):
        self.problem.reset()
        return self.optimizer.init_population(self.problem)

    def step(self, action: Any):
        return self.optimizer.update(action, self.problem)