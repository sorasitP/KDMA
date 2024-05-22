# Multi-Agent Navigation in Obstacle Environment using Knowledge Distillation 

This is project of deep reinforcement learning class in KMUTT university, this project inspired multi-agent navigation framework from this paper **Human-Inspired Multi-Agent Navigation using Knowledge Distillation**. [[arXiv](https://arxiv.org/abs/2103.10000)][[Youtube](https://youtu.be/tMctyEw8kRI)]. We try to add obstacles into multi-agent environment by adding the obstacle observation in agent's state which using same layer as neighbor observation in neural network policy.

## Dependencies
For dependencies and how to use multi-agent navigation framework, you can follow the tutorial from original KDMA github [[KDMA](https://github.com/xupei0610/KDMA)]

## Obstacle Environment
We inherit corridor scenario from KDMA framework where agents randomly spawn in both side of corridor and their goal is in opposite side. We add obstacle between agent's spawn point and their goal, The agents must navigate around obstacles instead of run straight ahead to goal and also avoid the other agents as the same time.
**INSERT GIF**


## Code Usage

About how to train the policy, you can follow the tutorial from original KDMA github [[KDMA](https://github.com/xupei0610/KDMA)]
In this part we show just about how we add obstacle observation into agent's state.

### Obstacle Observation
We have 2 types of Obstacle Observation (You can select each type by changing parameter in **config.py**)
-   Only nearest obstacle point where we input only the nearest point of each obstacle into agent's state. (To use this type, change <span style="color: blue;">ONLY_NEAREST=True</span>)
-   Obstacle points around the agent where we input desired number of obstacle points around agent into agent's state. (To use this type, change <span style="color: blue;">ONLY_NEAREST=False</span> and specify the desired number of points <span style="color: blue;">OBSEREVD_POINTS=60</span> **Maximum is 360**)


## Citation
    @inproceedings{kdma,
        author={Xu, Pei and Karamouzas, Ioannis},
        booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
        title={Human-Inspired Multi-Agent Navigation using Knowledge Distillation}, 
        year={2021},
        volume={},
        number={},
        pages={8105-8112},
        doi={10.1109/IROS51168.2021.9636463}
    }
