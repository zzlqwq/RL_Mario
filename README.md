# RL_Mario

## 项目简介

尝试用基于值函数逼近的强化学习方法玩经典的马里奥游戏，取得了一定成果。后续会对算法继续优化，以便计算设备不太行的玩家也能感受强化学习的魅力。

## 项目环境

- gym-super-mario-bros==7.3.0
- pytorch==1.9.0
- 其余所需环境使用conda或者pip下载最新版本即可

## 项目启动方法

直接运行main.py即可

## 项目组织结构

项目组织结构十分简单，五个python文件
1. EnvWrappers(定义了对于env环境的处理函数)
2. MarioNet.py(定义了卷积神经网络结构)
3. MetricLogger.py(定义了打印信息的函数)
4. Mario.py(集成了强化学习算法需要的各个模块)
5. main.py(定义了学习算法的训练过程以及检验训练效果的执行过程)

##  一些发现

1. 我训练了十万回合的马里奥，发现奖励确实上升了，但是也没有训练出来一个能通关的马里奥，我觉得可能是因为奖励设置的问题。
2. 我训练出来了四十多个网络，选取了两个效果还算可以的训练好的网络上传到了仓库。

## TODO

我的代码是基于pytorch官方改的，在我个人看来还有很大的优化空间
1. 可以将神经网络模型改成Duel Network
2. 受限于显卡内存我的经验池大小只能开到两万五千，可能会影响随机梯度下降的效果，我可以使用算法优化一下，毕竟很多玩家都没有那么大的内存吧。
3. 使用优先回放算法改进一下
4. 修改一下该环境的奖励，尝试让马里奥学习如何跑的更远。
