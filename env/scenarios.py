from typing import Callable, Sequence, Tuple
from .agents.base_agent import BaseAgent

import numpy
import math

__all__ = [
    "CompositeScenarios",
    "CircleCrossingScenario",
    "SquareCrossingScenario",

    "CircleCrossing6Scenario",
    "CircleCrossing12Scenario",
    "StaticWallScenario"
]

class BaseScenario():
    def __init__(self, seed: int=None):
        self.rng = numpy.random.RandomState()
        self.seed(seed)

        self.wall = [dict(start_point = (7,2), end_point =(10,1),wall_type='rec',velocity=(0,0),thickness=0)]
        self.has_wall = False
        ## wall should be dict of component to build rectangle wall,
        ## it consists of start_point (s:top-left) and end_point (e:bottom-right)
        ## thickness is only inside wall
        ## EXAMPLE: wall = dict(start_point = (7,2), end_point =(10,1))
        #############################
        #...........................#
        #.......swww................#
        #.......wwwe................#
        #...........................#
        #############################

    def spawn(self):
        raise NotImplementedError

    def seed(self, s: int=None):
        self.custom_seed = s is not None
        self.rng.seed(s)

    def collide(self, agent0: BaseAgent, agent1: BaseAgent):
        if agent0.visible and agent1.visible:
            dist2 = (agent0.position.x-agent1.position.x)**2 + (agent0.position.y-agent1.position.y)**2
            return dist2 <= (agent0.radius+agent1.radius)**2
        return False
    
    def wall_collide(self, agent: BaseAgent):
        if agent.visible:
            for w in self.wall:    
                if w['wall_type'] == 'rectangle':
                    x_min = w['start_point'][0]
                    y_min = w['end_point'][1]
                    x_max = w['end_point'][0]
                    y_max = w['start_point'][1]
                    list_dxy = self.find_closet_rec_point((agent.position.x,agent.position.y),(x_min,y_min),(x_max,y_max),w['thickness'])
                    
                    for (dx,dy) in list_dxy:
                        if (dx**2 + dy**2) <= agent.radius*agent.radius:
                            return True
                elif w['wall_type'] == 'circle':
                    center_x = w['start_point'][0]
                    center_y = w['start_point'][1]
                    circle_r = w['radius']
                    v2c_x = center_x - agent.position.x
                    v2c_y = center_y - agent.position.y
                    dist2c = (v2c_x**2 + v2c_y**2)**0.5
                    if dist2c < circle_r-w['thickness']:
                        dist2r = dist2c - (circle_r-w['thickness'])
                    elif dist2c >= circle_r:
                        dist2r = dist2c - circle_r
                    else:
                        dist2r = 0
                    dx = v2c_x/dist2c * dist2r
                    dy = v2c_y/dist2c * dist2r
                    
                    if (dx**2 + dy**2) <= agent.radius*agent.radius:
                        return True
        return False

    def search_360_wall(self,agent: BaseAgent):
        wall_list = []
        theta_list = []
        if agent.visible:
            r = agent.radius
            ob_r = agent.observe_radius
            agent_p = (agent.position.x,agent.position.y)
            for degree in range(0,360,6): # loop for 360 degree
                theta  = math.pi/180 * degree
                ## find the point with that degree
                p_x = (r * math.cos(theta)) + agent_p[0]
                p_y = (r * math.sin(theta)) + agent_p[1]
                dy = (p_y - agent_p[1])
                dx = (p_x - agent_p[0])
                ## find linear equation
                if degree==90 or degree ==270:
                    m=1
                    c =  (m*p_x)
                else:
                    m  = dy / dx
                    c = p_y - (m*p_x)
                ## find unit vector for that degree
                dist_p = (dx**2 +dy**2)**0.5
                unit_vector = (round(dx/dist_p,4),round(dy/dist_p,4))
                min_dist = ob_r  ## the point on wall must in observe radius
                wall_point = [0,0,0,0] ##
                for w in self.wall:
                    if w['wall_type'] == 'rectangle':
                        rec_min=(w['start_point'][0],w['end_point'][1])
                        rec_max=(w['end_point'][0],w['start_point'][1])
                        if degree != 90 or degree != 270:
                            for x in [rec_min[0],rec_max[0]]:
                                y = x*m+c
                                if y >= rec_min[1] and y <= rec_max[1]:
                                    _dx = x - agent_p[0]
                                    _dy = y - agent_p[1]
                                    _dist = (_dx**2 +_dy**2)**0.5
                                    _uv = (round(_dx/_dist,4),round(_dy/_dist,4))
                                    if _uv == unit_vector:
                                        if _dist <= min_dist:
                                            min_dist = _dist
                                            wall_point[0] = _dx
                                            wall_point[1] = _dy
                                            wall_point[2] = w['velocity'][0] - 0
                                            wall_point[3] = w['velocity'][1] - 0
                        for y in [rec_min[1],rec_max[1]]:
                            if m != 0:
                                if degree != 90 and degree != 270:
                                    x = (y-c)/m
                                else:
                                    x = c
                                if x >= rec_min[0] and x <= rec_max[0]:
                                    _dx = x - agent_p[0]
                                    _dy = y - agent_p[1]
                                    _dist = (_dx**2 +_dy**2)**0.5
                                    _uv = (round(_dx/_dist,4),round(_dy/_dist,4))
                                    if _uv == unit_vector:
                                        if _dist <= min_dist:
                                            min_dist = _dist
                                            wall_point[0] = _dx
                                            wall_point[1] = _dy
                                            wall_point[2] = w['velocity'][0] - 0
                                            wall_point[3] = w['velocity'][1] - 0
                if wall_point != [0,0,0,0]:
                    wall_list.append(wall_point)
                theta_list.append(theta)
        return wall_list,theta_list

    def nearest_wall(self,agent: BaseAgent):
        n = []
        if agent.visible:
            for w in self.wall:
                if w['wall_type'] == 'rectangle':
                    x_min = w['start_point'][0]
                    y_min = w['end_point'][1]
                    x_max = w['end_point'][0]
                    y_max = w['start_point'][1]
                    
                    vx = w['velocity'][0] - agent.velocity.x
                    vy = w['velocity'][1] - agent.velocity.y
                    list_dxy = self.find_closet_rec_point((agent.position.x,agent.position.y),(x_min,y_min),(x_max,y_max),w['thickness'])                
                    for (dx,dy) in list_dxy:
                        if (dx**2 + dy**2) <= agent.observe_radius*agent.observe_radius:
                            n.extend([dx,dy,vx,vy])                    
                elif w['wall_type'] == 'circle':
                    center_x = w['start_point'][0]
                    center_y = w['start_point'][1]
                    circle_r = w['radius']
                    v2c_x = center_x - agent.position.x
                    v2c_y = center_y - agent.position.y
                    dist2c = (v2c_x**2 + v2c_y**2)**0.5
                    if dist2c < circle_r-w['thickness']:
                        dist2r = dist2c - (circle_r-w['thickness'])
                    elif dist2c >= circle_r:
                        dist2r = dist2c - circle_r
                    else:
                        dist2r = 0
                    dx = v2c_x/dist2c * dist2r
                    dy = v2c_y/dist2c * dist2r
                    
                    if (dx**2 + dy**2) <= agent.observe_radius*agent.observe_radius:
                        vx = w['velocity'][0] - agent.velocity.x
                        vy = w['velocity'][1] - agent.velocity.y
                        n.extend([dx,dy,vx,vy]) 
                        
        return n

    def find_closet_rec_point(self, agent_pos,rec_min,rec_max,thickness=0):
        def nearest_point(p,min,max,in_offset):
            if p < min:
                nearest_p = min
            elif p> max:
                nearest_p = max
            else:
                nearest_p = min+in_offset if p - min < max - p else max-in_offset
                return nearest_p, True
            return nearest_p, False

        wall_x, in_x = nearest_point(agent_pos[0],rec_min[0],rec_max[0],thickness)
        wall_y, in_y = nearest_point(agent_pos[1],rec_min[1],rec_max[1],thickness)
        dx = wall_x - agent_pos[0]
        dy = wall_y - agent_pos[1]
        if in_x and in_y:
            if abs(dx)<= thickness or abs(dy)<= thickness:
                return [(0,0)]
            if abs(dx) == abs(dy):
                return [(dx,0),(0,dy)]
            elif abs(dx) < abs(dy):
                dy =0
            else:
                dx =0
        elif in_x:
            dx = 0
        elif in_y:
            dy = 0
        return [(dx,dy)]

    def placeable(self, agent: BaseAgent):
        for o in self.agents:
            if self.collide(agent, o):
                return False
        return True

    def __iter__(self):
        self.counter = 0
        self.agents = []
        return self
    
    def __next__(self):
        agent = self.spawn()
        self.agents.append(agent)
        self.counter += 1
        return agent


class CompositeScenarios(BaseScenario):

    def __init__(self, scenarios: Sequence[BaseScenario], prob=None, seed: int=None):
        self.scenarios = [s for s in scenarios]
        self.prob = prob
        super().__init__(seed)
    
    def seed(self, seed):
        super().seed(seed)
        for i, s in enumerate(self.scenarios):
            if not s.custom_seed:
                s.rng = self.rng
            
    def __iter__(self):
        i = self.rng.choice(len(self.scenarios), p=self.prob)
        self.spawn = self.scenarios[i].spawn
        return self.scenarios[i].__iter__()


class CircleCrossingScenario(BaseScenario):

    def __init__(self,
        n_agents: int or Tuple[int, int],
        radius: float or Tuple[float, float],
        agent_wrapper: Callable[[], BaseAgent],
        noise: float = 0,
        min_distance: float = 0,
        seed: int = None
    ):
        self.n_agents = n_agents
        self.radius = radius
        self.agent_wrapper = agent_wrapper
        self.min_distance = min_distance
        self.noise = noise
        super().__init__(seed)
    
    def __iter__(self):
        if hasattr(self.n_agents, "__len__"):
            self._n_agents = self.rng.randint(self.n_agents[0], self.n_agents[1])
        else:
            self._n_agents = self.n_agents
        if hasattr(self.radius, "__len__"):
            self._radius = self.rng.random()*(self.radius[1]-self.radius[0]) + self.radius[0]
        else:
            self._radius = self.radius
        return super().__iter__()

    def spawn(self):
        if self.counter >= self._n_agents:
            raise StopIteration

        agent = self.agent_wrapper()
        r = agent.radius
        agent.radius += self.min_distance

        while True:
            a = self.rng.random() * 2*numpy.pi
            agent.position = numpy.cos(a)*self._radius, numpy.sin(a)*self._radius
            if self.noise:
                agent.position = (
                    agent.position.x + (self.rng.random()-0.5)*2*self.noise,
                    agent.position.y + (self.rng.random()-0.5)*2*self.noise
                )
            if self.placeable(agent): break
        agent.goal = -agent.position.x, -agent.position.y
        # agent.velocity = agent.preferred_velocity(0.12)
        agent.radius = r
        return agent


class SquareCrossingScenario(BaseScenario):
    def __init__(self,
        n_agents: int or Tuple[int, int],
        width: float or Tuple[float, float],
        height: float or Tuple[float, float],
        vertical: bool,
        horizontal: bool,
        agent_wrapper: Callable[[], BaseAgent],
        noise: float = 0,
        min_distance: float = 0,
        seed: int = None
    ):
        self.n_agents = n_agents
        self.width = width
        self.height = height
        self.vertical = vertical
        self.horizontal = horizontal
        self.agent_wrapper = agent_wrapper
        self.min_distance = min_distance
        super().__init__(seed)

    def __iter__(self):
        if hasattr(self.n_agents, "__len__"):
            self._n_agents = self.rng.randint(self.n_agents[0], self.n_agents[1])
        else:
            self._n_agents = self.n_agents
        if hasattr(self.width, "__len__"):
            self._width = self.rng.random()*(self.width[1]-self.width[0]) + self.width[0]
        else:
            self._width = self.width
        if hasattr(self.height, "__len__"):
            self._height = self.rng.random()*(self.height[1]-self.height[0]) + self.height[0]
        else:
            self._height = self.height
        self.goals = []
        return super().__iter__()
    

    def spawn(self):
        if self.counter >= self._n_agents:
            raise StopIteration

        agent = self.agent_wrapper()
        r = agent.radius
        agent.radius += self.min_distance
        r2 = agent.radius*agent.radius

        if self.vertical and self.horizontal:
            vertical = self.rng.random() > 0.5
        else:
            vertical = self.vertical
        while True:
            if vertical:
                x = self.rng.random()-0.5
                if self.horizontal: x *= 0.5
                y = (self.rng.random()-0.5) * 0.5 # (-0.25, 0.25)
                if y < 0:
                    y -= 0.25
                else:
                    y += 0.25
            else:
                x = (self.rng.random()-0.5) * 0.5
                if x < 0:
                    x -= 0.25
                else:
                    x += 0.25
                y = self.rng.random()-0.5
                if self.vertical: y *= 0.5
            agent.position = x * self._width, y * self._height
            if self.placeable(agent): break
        while True:
            if vertical:
                x = self.rng.random()-0.5
                if self.horizontal: x *= 0.5
                y = (self.rng.random()-0.5) * 0.5 # (-0.25, 0.25)
                if y < 0:
                    y -= 0.25
                else:
                    y += 0.25
                if ((agent.position.y > 0 and y > 0) or (agent.position.y < 0 and y < 0)):
                    y = -y
            else:
                x = (self.rng.random()-0.5) * 0.5
                if x < 0:
                    x -= 0.25
                else:
                    x += 0.25
                if (agent.position.x > 0 and x > 0) or (agent.position.x < 0 and x < 0):
                    x = -x
                y = self.rng.random()-0.5
                if self.vertical: y *= 0.5
            x *= self._width
            y *= self._height
            if (agent.position.x-x)**2 + (agent.position.y-y)**2 <= r2:
                continue
            placeable = True
            for gx, gy in self.goals:
                if (gx-x)**2 + (gy-y)**2 <= r2:
                    placeable = False
                    break
            if placeable:
                agent.goal = x, y
                break
        self.goals.append((agent.goal.x, agent.goal.y))
        agent.radius = r
        return agent



class PredefinedScenario(BaseScenario):
    def __init__(self,
        agent_wrapper: Callable[[], BaseAgent],
        scale : int = 1
    ):
        self.agent_wrapper = agent_wrapper
        super().__init__()
        self.scale = scale
    def spawn(self):
        if self.counter >= len(self.POS):
            raise StopIteration

        agent = self.agent_wrapper()
        agent.position = self.POS[self.counter][0]*self.scale, self.POS[self.counter][1]*self.scale
        agent.goal = self.GOAL[self.counter][0]*self.scale, self.GOAL[self.counter][1]*self.scale
        return agent

class CircleCrossing6Scenario(PredefinedScenario):

    POS = (
        ( 3.7104692, -1.5660178 ),
        ( 2.4903586, 3.3488475 ),
        ( -1.2998195, 3.90333 ),
        ( -4.0630183, 1.1232517 ),
        ( -2.9886541, -2.7252258 ),
        ( 0.58739246, -4.0228777 )
    )

    GOAL = (
        ( -3.9725885, 1.7843779 ),
        ( -2.7661666, -3.4184108 ),
        ( 1.4674827, -4.0561664 ),
        ( 4.2410774, -0.92563815 ),
        ( 3.1549813, 3.1627561 ),
        ( -1.0496976, 4.6132372 )
    )

    def __init__(self,
        agent_wrapper: Callable[[], BaseAgent],
        scale : int = 0.46716322317696163
    ):
        super().__init__(agent_wrapper, scale)
    


class CircleCrossing12Scenario(PredefinedScenario):

    POS = (
        (3.6143744, 1.1175725),
        (2.3853909, 2.2851516),
        (1.2247712, 3.8650695),
        (-0.4762151, 3.3848816),
        (-2.838231, 2.6234833),
        (-3.6877534, 0.9764592),
        (-3.396552, -1.4452063),
        (-2.0577695, -2.4942946),
        (-0.62386751, -4.0112653),
        (1.1290323, -3.4432024),
        (2.4359613, -2.1397746),
        (3.4712598, -0.83114835)
    )

    GOAL = (
        (-4.2847164, -1.5056181),
        (-3.2118745, -2.9733542),
        (-1.0373271, -3.7547732),
        (0.65222115, -4.2867512),
        (3.2604236, -2.8074883),
        (4.221533, -0.909147),
        (3.7665522, 1.9906672),
        (2.6808445, 3.550647),
        (1.2148837, 4.2657175),
        (-1.5079789, 4.1998218),
        (-3.1926017, 3.1203148),
        (-4.2111265, 1.0495945)
    )

    def __init__(self,
        agent_wrapper: Callable[[], BaseAgent],
        scale : int = 0.47624522851691475
    ):
        super().__init__(agent_wrapper, scale)
        
########### Add Wall scenario ############
class StaticWallScenario(BaseScenario):
    def __init__(self,
        n_agents: int or Tuple[int, int],
        width: float or Tuple[float, float],
        height: float or Tuple[float, float],
        vertical: bool,
        horizontal: bool,
        agent_wrapper: Callable[[], BaseAgent],
        wall: list,
        noise: float = 0,
        min_distance: float = 0,
        seed: int = None,
    ):
        
        ## wall should be dict of component to build rectangle wall,
        ## it consists of start_point (s:top-left) and end_point (e:bottom-right)
        ## EXAMPLE: wall = dict(start_point = (7,2), end_point =(10,1))
        #############################
        #...........................#
        #.......swww................#
        #.......wwwe................#
        #...........................#
        #############################

        self.n_agents = n_agents
        self.width = width
        self.height = height
        self.vertical = vertical
        self.horizontal = horizontal
        self.agent_wrapper = agent_wrapper
        self.min_distance = min_distance
        super().__init__(seed)

        self.has_wall = True
        self.dynamic = False
        self.wall = wall

    def __iter__(self):
        if hasattr(self.n_agents, "__len__"):
            self._n_agents = self.rng.randint(self.n_agents[0], self.n_agents[1])
        else:
            self._n_agents = self.n_agents
        if hasattr(self.width, "__len__"):
            self._width = self.rng.random()*(self.width[1]-self.width[0]) + self.width[0]
        else:
            self._width = self.width
        if hasattr(self.height, "__len__"):
            self._height = self.rng.random()*(self.height[1]-self.height[0]) + self.height[0]
        else:
            self._height = self.height
        self.goals = []
        self.has_wall = True
        return super().__iter__()
    

    def spawn(self):
        if self.counter >= self._n_agents:
            raise StopIteration

        agent = self.agent_wrapper()
        r = agent.radius
        agent.radius += self.min_distance
        r2 = agent.radius*agent.radius

        if self.vertical and self.horizontal:
            vertical = self.rng.random() > 0.5
        else:
            vertical = self.vertical
        while True:
            if vertical:
                x = self.rng.random()-0.5
                if self.horizontal: x *= 0.5
                y = (self.rng.random()-0.5) * 0.5 # (-0.25, 0.25)
                if y < 0:
                    y -= 0.25
                else:
                    y += 0.25
            else:
                x = (self.rng.random()-0.5) * 0.5
                if x < 0:
                    x -= 0.25
                else:
                    x += 0.25
                y = self.rng.random()-0.5
                if self.vertical: y *= 0.5
            agent.position = x * self._width, y * self._height
            if self.placeable(agent): break
        while True:
            if vertical:
                x = self.rng.random()-0.5
                if self.horizontal: x *= 0.5
                y = (self.rng.random()-0.5) * 0.5 # (-0.25, 0.25)
                if y < 0:
                    y -= 0.25
                else:
                    y += 0.25
                if ((agent.position.y > 0 and y > 0) or (agent.position.y < 0 and y < 0)):
                    y = -y
            else:
                x = (self.rng.random()-0.5) * 0.5
                if x < 0:
                    x -= 0.25
                else:
                    x += 0.25
                if (agent.position.x > 0 and x > 0) or (agent.position.x < 0 and x < 0):
                    x = -x
                y = self.rng.random()-0.5
                if self.vertical: y *= 0.5
            x *= self._width
            y *= self._height
            if (agent.position.x-x)**2 + (agent.position.y-y)**2 <= r2:
                continue
            placeable = True
            for gx, gy in self.goals:
                if (gx-x)**2 + (gy-y)**2 <= r2:
                    placeable = False
                    break
            if placeable:
                agent.goal = x, y
                break
        self.goals.append((agent.goal.x, agent.goal.y))
        agent.radius = r
        return agent
    
    def visualize(self,fig,ax):
        import matplotlib.pyplot as plt

        fig = fig
        ax = ax
        for w in self.wall:
            x1 = numpy.array([w['start_point'][0],w['start_point'][0],w['end_point'][0],w['end_point'][0],w['start_point'][0]])
            y1 = numpy.array([w['end_point'][1],w['start_point'][1],w['start_point'][1],w['end_point'][1],w['end_point'][1]])
            plt.plot(x1, y1,'r-')