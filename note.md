# 一些笔记

## 问题（数据集相关）

- 问题抽象类为 basic problem
- 子问题每一个都继承了 basic problem
- 子问题的 y（即目标函数）由 func（）返回，e.g.蛋白质的目标函数为 energy 变量
- bbob.py 和 bbob_torch.py 的区别应该是是否使用了 torch，有一个似乎只使用了 numpy
- 子问题应当有三类 1. 不带噪声的 2.带噪声的 3.蛋白质

## 解决方案

- 先写个pso先（保险）
- 能搞点rl+llm就好玩了
- 能搞出mcrs更好玩
