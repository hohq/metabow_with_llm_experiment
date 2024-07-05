import torch 
import pickle
from config import get_config
from optimizer import PSO,Particle
from utils import *
from tqdm import tqdm
from problem import protein_docking
from PBO import PBO_Env
if __name__ == '__main__':
    config_m=get_config()
    # print("hello world")
    train_set=construct_problem_set(config_m)
    # print (train_set)
    torch.set_grad_enabled(True)
    dim=config_m.dim
    
    for problem_id, problem in enumerate(train_set):
        env = PBO_Env(problem, PSO(particles_num=10,max_p=100,max_v=100,dimension=dim,C1=2,C2=2,W_max=0.9,W_min=0.4,epoches=100))
        print(f'problem_id: {problem_id}')
        print(f'problem: {problem}')
        # dim=problem.dim
        print(f'problem.dim: {dim}')
        # print(f'problem.data: {problem.data}')
        print(f'problem.batch_size: {problem.batch_size}')
        print(f'problem.N: {problem.N}')
        # print(f'problem.ptr: {problem.ptr}')
        # print(f'problem.index: {problem.index}')
        # print(f'problem.get_datasets: {problem.get_datasets}')
        # train=
        # print(f'train: {train}')
        
        result = []  # 结果列表
        
            
            

