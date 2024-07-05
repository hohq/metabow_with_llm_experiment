import numpy as np
from problem.basic_problem import Basic_Problem
from problem import Protein_Docking
import time
from optimizer_template import optimizer_template
import torch

class Particle:
    def __init__(self,max_p,max_v,dimension):
        self.position = np.random.uniform(-max_p,max_p,dimension)
        self.velocity = np.random.uniform(-max_v,max_v,dimension)
        self.best_position = self.position
        self.best_value = np.inf

    def get_position(self):
        return self.position
    
    def get_velocity(self):
        return self.velocity
    
    def get_best_position(self):
        return self.best_position
    
    def get_best_value(self):
        return self.best_value
    
    def set_position(self,position):
        self.position = position
        
    def set_velocity(self,velocity):
        self.velocity = velocity
        
    
class PSO(optimizer_template):
    def __init__(self,particles_num,max_p,max_v,dimension,C1,C2,W_max,W_min,epoches,problem):
        self.c1=C1
        self.c2=C2
        self.w=W_max
        self.dimension=dimension
        self.particles_num=particles_num
        self.max_p=max_p
        self.max_v=max_v
        self.epoches=epoches
        self.w_max=W_max
        self.w_min=W_min
        self.particles = [Particle(self.max_p,self.max_v,dimension) for i in range(self.particles_num)]
        self.epoch_now=0
        self.global_best_position = np.zeros(dimension)
        self.global_best_value = np.inf
        # self.func= func
        self.problem=problem
        
    def func(self,problem,position):
        return problem.eval(position)
    
    def get_global_best_position(self):
        return self.global_best_position
    
    def get_global_best_value(self):
        return self.global_best_value
    
    def set_global_best_value(self,value):
        self.global_best_value = value
        
    def set_global_best_position(self,position):
        self.global_best_position = position
        
    def update_v(self,particle):
        particle.velocity = self.w*particle.velocity + self.c1*np.random.rand()*(particle.best_position-particle.position) + self.c2*np.random.rand()*(self.global_best_position-particle.position)

    def update_p(self,particle):
        particle.position = particle.position + particle.velocity
        
    def update(self,epoch_now=1):
        for particle in self.particles:
            self.update_v(particle)
            self.update_p(particle)
            
            epoch_now+=1
            self.w=self.w_max-(self.w_max-self.w_min)*(self.epoch_now/self.epoches)
            print(particle.position)
            
            # value = self.func([-26.7622428,-44.85113596, 59.94787813,  52.44768341 , 18.58916244, 78.71874367, -46.85806894, -44.73517205,  22.98333243, -43.42504494, -65.29808822 , 25.77216542])
            value = self.func(self.problem,particle.position)
            
            if value < particle.best_value:
                particle.best_value = value
                particle.best_position = particle.position
            if value < self.global_best_value:
                self.global_best_value = value
                self.global_best_position = particle.position
                

        
            