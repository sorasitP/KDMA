from functools import partial
import os, time
import torch

from env.scenarios import *
from models.networks import ExpertNetwork
from models.env import Env
from models.agent import DLAgent

import config

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default='pretrain/policy/ckpt') #default=None
parser.add_argument("--max_trials", type=int, default=10)
parser.add_argument("--scene", type=str, default="24-static_wall",
    choices=["6-circle", "12-circle", "20-circle", "24-circle", "20-corridor", "24-corridor", "20-square", "24-square","24-static_wall","24-dynamic_wall"]
)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--view", type=str, default="True")
settings = parser.parse_args()

if settings.view == "True":
        view_param = True
else:
    view_param = False

def env_wrapper(model, expert=None):
    agent_wrapper = partial(DLAgent,
        preferred_speed=config.PREFERRED_SPEED, max_speed=config.MAX_SPEED,
        radius=config.AGENT_RADIUS, observe_radius=config.NEIGHBORHOOD_RADIUS,
        expert=expert, model=model
        )
    if settings.scene == "6-circle": 
        scenario = CircleCrossing6Scenario(agent_wrapper=agent_wrapper)
    elif settings.scene == "12-circle":
        scenario = CircleCrossing12Scenario(agent_wrapper=agent_wrapper)
    elif settings.scene == "20-circle":
        scenario = CircleCrossingScenario(n_agents=20, agent_wrapper=agent_wrapper, min_distance=0.3, radius=4)
    elif settings.scene == "24-circle":
        scenario = CircleCrossingScenario(n_agents=24, agent_wrapper=agent_wrapper, min_distance=0.3, radius=4)
    elif settings.scene == "20-square":
        scenario = SquareCrossingScenario(n_agents=20, agent_wrapper=agent_wrapper, min_distance=0.3, vertical=True, horizontal=True, width=8, height=8)
    elif settings.scene == "24-square":
        scenario = SquareCrossingScenario(n_agents=24, agent_wrapper=agent_wrapper, min_distance=0.3, vertical=True, horizontal=True, width=8, height=8)
    elif settings.scene == "20-corridor":
        scenario = CompositeScenarios([
            SquareCrossingScenario(n_agents=20, agent_wrapper=agent_wrapper, min_distance=0.3, vertical=True, horizontal=False, width=8, height=8),
            SquareCrossingScenario(n_agents=20, agent_wrapper=agent_wrapper, min_distance=0.3, vertical=False, horizontal=True, width=8, height=8)
        ])
    elif settings.scene == "24-corridor":
        scenario = CompositeScenarios([
            SquareCrossingScenario(n_agents=24, agent_wrapper=agent_wrapper, min_distance=0.3, vertical=True, horizontal=False, width=8, height=8),
            SquareCrossingScenario(n_agents=24, agent_wrapper=agent_wrapper, min_distance=0.3, vertical=False, horizontal=True, width=8, height=8)
        ])    
    elif settings.scene == "24-static_wall":
        scenario = StaticWallScenario(n_agents=20, agent_wrapper=agent_wrapper, min_distance=0.3, vertical=True, horizontal=False, width=8, height=8, 
                                      wall=[dict(start_point = (-5,5), end_point =(5,-5),wall_type='rectangle',velocity=(0,0),thickness=0),
                                            dict(start_point = (-5,0.5), end_point =(1,-0.5),wall_type='rectangle',velocity=(0,0),thickness=0)]) 
    # elif settings.scene == "24-dynamic_wall":
    #     scenario = DynamicWallScenario(n_agents=20, agent_wrapper=agent_wrapper, min_distance=0.3, vertical=True, horizontal=False, width=8, height=8,
    #                                    wall=[dict(start_point = (-5,5), end_point =(5,-5),wall_type='rectangle',velocity=(1,0),thickness=0),
    #                                         dict(start_point = (-5,0.5), end_point =(1,-0.5),wall_type='rectangle',velocity=(10,0),thickness=0)])
              
    else:
        raise ValueError("Unrecognized scene: {}".format(settings.scene))

    env = Env(scenario=scenario, fps=1./config.STEP_TIME, timeout=config.VISUALIZATION_TIMEOUT, frame_skip=config.FRAME_SKIP,
        view=view_param
    )
    return env

def evaluate(ckpt_file):
    print(ckpt_file)
    print(settings.scene)
    # get env from learned model #
    ckpt = torch.load(ckpt_file, map_location="cpu")
    state_dict = {}
    for k, v in ckpt["model"].items():
        if "model.actor.model." in k:
            state_dict[k[18:]] = v
    # load expert model
    model = ExpertNetwork(agent_dim=3, neighbor_dim=4, out_dim=2)
    model.load_state_dict(state_dict)
    if settings.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = settings.device
    model.to(device)

    env = env_wrapper(model)
    env.seed(0)
    model.eval()

    done, info = True, None
    trials = 0
    total, lifetime, arrived, collided = 0, 0, 0, 0
    avg_reward, speed = [], []
    episode = 0
    while True:
        if done:
            state = env.reset()
            reward = [[] for _ in range(len(env.agents))]
            if view_param:
                env.figure.axes.set_title(os.path.join(os.path.basename(os.path.dirname(ckpt_file)), os.path.basename(ckpt_file)))
            t = time.time()
        else:
            state = state_
        act = [ag.act(s, env) for ag, s in zip(env.agents, state)]

        state_, rews, done, info = env.step(act)
        delay = config.STEP_TIME - time.time() + t
        if delay > 0:
            time.sleep(delay)
        t = time.time()

        for idx, (_, r, ag) in enumerate(zip(state, rews, env.agents)):
            if _ is not None:
                reward[idx].append(r)
                speed.append((ag.velocity.x**2 + ag.velocity.y**2)**0.5)
                lifetime += 1
        if done:
            episode+=1
            print("episode ",episode," finish!")
            arrived += len(info["arrived_agents"])
            collided += len(info["collided_agents"])
            total += len(env.agents)
            rews = []
            for r in reward:
                rews.append(r[-1])
                for _ in reversed(r[:-1]):
                    rews.append(_ + 0.99*rews[-1])
            avg_reward.append(sum(rews)/len(rews))

            trials += 1
            time.sleep(2)

            if trials >= settings.max_trials:
                success_rate = arrived/total
                collision_rate = collided/total
                avg_reward = sum(avg_reward)/len(avg_reward)
                avg_time = lifetime/total
                print("[PERFORM] Collision: {:.2f}, Success: {:.2f}, Reward: {:.2f}, Avg Time: {:.2f}, {}".format(
                    collision_rate, success_rate, avg_reward, avg_time,
                    time.strftime("%m-%d %H:%M:%S")
                ))
                total, lifetime, arrived, collided = 0, 0, 0, 0
                avg_reward, speed = [], []
                break


if __name__ == "__main__":
    if os.path.isfile(settings.ckpt):
        evaluate(settings.ckpt)
    else:
        def check(path):
            for f in sorted(os.listdir(path)):
                filename = os.path.join(path, f)
                if "ckpt" == f:
                    evaluate(filename)
                elif os.path.isdir(filename):
                    check(filename)
        check(settings.ckpt)
