"""
.. module:: pinball
   :platform: Unix, Windows
   :synopsis: Pinball domain for reinforcement learning

.. moduleauthor:: Pierre-Luc Bacon <pierrelucbacon@gmail.com>
.. adaptedby:: Luiz Alfredo Thomasini <luizthomasini@gmail.com>
"""

import random
import argparse, os
import numpy as np
from itertools import *

try:
    import pygame
except ImportError as e:
    print('Pygame not available', e)

try:
    import gymnasium as gym
except ImportError as e:
    print('Gymnasium not available', e)

class BallModel:
    """ This class maintains the state of the ball
    in the pinball domain. It takes care of moving
    it according to the current velocity and drag coefficient.
    """
    DRAG = 0.995

    def __init__(self, start_position, radius):
        """
        :param start_position: The initial position
        :type start_position: float
        :param radius: The ball radius
        :type radius: float
        """
        self.position = start_position
        self.radius = radius
        self.xdot = 0.0
        self.ydot = 0.0

    def add_impulse(self, delta_xdot, delta_ydot):
        """ Change the momentum of the ball
        :param delta_xdot: The change in velocity in the x direction
        :type delta_xdot: float
        :param delta_ydot: The change in velocity in the y direction
        :type delta_ydot: float
        """
        self.xdot += delta_xdot/5.0
        self.ydot += delta_ydot/5.0
        self._clip(self.xdot)
        self._clip(self.ydot)

    def add_drag(self):
        """ Add a fixed amount of drag to the current velocity """
        self.xdot *= self.DRAG
        self.ydot *= self.DRAG

    def step(self):
        """ Move the ball by one increment """
        self.position[0] += self.xdot*self.radius/20.0
        self.position[1] += self.ydot*self.radius/20.0

    def _clip(self, val, low=-1, high=1):
        """ Clip a value in a given range """
        if val > high:
            val = high
        if val < low:
            val = low
        return val

class PinballObstacle:
    """ This class represents a single polygon obstacle in the
    pinball domain and detects when a :class:`BallModel` hits it.

    When a collision is detected, it also provides a way to
    compute the appropriate effect to apply on the ball.
    """
    def __init__(self, points):
        """
        :param points: A list of points defining the polygon
        :type points: list of lists
        """
        self.points = points
        self.min_x = min(self.points, key=lambda pt: pt[0])[0]
        self.max_x = max(self.points, key=lambda pt: pt[0])[0]
        self.min_y = min(self.points, key=lambda pt: pt[1])[1]
        self.max_y = max(self.points, key=lambda pt: pt[1])[1]

        self._double_collision = False
        self._intercept = None

    def collision(self, ball):
        """ Determines if the ball hits this obstacle

        :param ball: An instance of :class:`BallModel`
        :type ball: :class:`BallModel`
        """
        self._double_collision = False

        if ball.position[0] - ball.radius > self.max_x:
            return False
        if ball.position[0] + ball.radius < self.min_x:
            return False
        if ball.position[1] - ball.radius > self.max_y:
            return False
        if ball.position[1] + ball.radius < self.min_y:
            return False

        a, b = tee(np.vstack([np.array(self.points), self.points[0]]))
        next(b, None)
        intercept_found = False
        for pt_pair in zip(a, b):
            if self._intercept_edge(pt_pair, ball):
                if intercept_found:
                    # Ball has hit a corner
                    self._intercept = self._select_edge(pt_pair, self._intercept, ball)
                    self._double_collision = True
                else:
                    self._intercept = pt_pair
                    intercept_found = True

        return intercept_found

    def collision_effect(self, ball):
        """ Based of the collision detection result triggered
        in :func:`PinballObstacle.collision`, compute the
            change in velocity.

        :param ball: An instance of :class:`BallModel`
        :type ball: :class:`BallModel`
        """
        if self._double_collision:
            return [-ball.xdot, -ball.ydot]

        # Normalize direction
        obstacle_vector = self._intercept[1] - self._intercept[0]
        if obstacle_vector[0] < 0:
            obstacle_vector = self._intercept[0] - self._intercept[1]

        velocity_vector = np.array([ball.xdot, ball.ydot])
        theta = self._angle(velocity_vector, obstacle_vector) - np.pi
        if theta < 0:
            theta += 2*np.pi

        intercept_theta = self._angle([-1, 0], obstacle_vector)
        theta += intercept_theta

        if theta > 2*np.pi:
            theta -= 2*np.pi

        velocity = np.linalg.norm([ball.xdot, ball.ydot])

        return [velocity*np.cos(theta), velocity*np.sin(theta)]

    def _select_edge(self, intersect1, intersect2, ball):
        """ If the ball hits a corner, select one of two edges.

        :param intersect1: A pair of points defining an edge of the polygon
        :type intersect1: list of lists
        :param intersect2: A pair of points defining an edge of the polygon
        :type intersect2: list of lists
        :returns: The edge with the smallest angle with the velocity vector
        :rtype: list of lists

        """
        velocity = np.array([ball.xdot, ball.ydot])
        obstacle_vector1 = intersect1[1] - intersect1[0]
        obstacle_vector2 = intersect2[1] - intersect2[0]

        angle1 = self._angle(velocity, obstacle_vector1)
        if angle1 > np.pi:
            angle1 -= np.pi

        angle2 = self._angle(velocity, obstacle_vector2)
        if angle1 > np.pi:
            angle2 -= np.pi

        if np.abs(angle1 - (np.pi/2.0)) < np.abs(angle2 - (np.pi/2.0)):
            return intersect1
        return intersect2

    def _angle(self, v1, v2):
        """ Compute the angle difference between two vectors

        :param v1: The x,y coordinates of the vector
        :type: v1: list
        :param v2: The x,y coordinates of the vector
        :type: v2: list
        :rtype: float

    	"""
        angle_diff = np.arctan2(v1[0], v1[1]) - np.arctan2(v2[0], v2[1])
        if angle_diff < 0:
            angle_diff += 2*np.pi
        return angle_diff

    def _intercept_edge(self, pt_pair, ball):
        """ Compute the projection on and edge and find out

        if it intercept with the ball.
        :param pt_pair: The pair of points defining an edge
        :type pt_pair: list of lists
        :param ball: An instance of :class:`BallModel`
        :type ball: :class:`BallModel`
        :returns: True if the ball has hit an edge of the polygon
        :rtype: bool

        """
        # Find the projection on an edge
        obstacle_edge = pt_pair[1] - pt_pair[0]
        difference = np.array(ball.position) - pt_pair[0]

        scalar_proj = difference.dot(obstacle_edge)/obstacle_edge.dot(obstacle_edge)
        if scalar_proj > 1.0:
            scalar_proj = 1.0
        elif scalar_proj < 0.0:
            scalar_proj = 0.0

        # Compute the distance to the closest point
        closest_pt = pt_pair[0] + obstacle_edge*scalar_proj
        obstacle_to_ball = ball.position - closest_pt
        distance = obstacle_to_ball.dot(obstacle_to_ball)

        if distance <= ball.radius*ball.radius:
            # A collision only if the ball is not already moving away
            velocity = np.array([ball.xdot, ball.ydot])
            ball_to_obstacle  = closest_pt - ball.position

            angle = self._angle(ball_to_obstacle, velocity)
            if angle > np.pi:
                angle = 2*np.pi - angle

            if angle > np.pi/1.99:
                return False

            return True
        else:
            return False


class PinballModel:
    """ This class is a self-contained model of the pinball
    domain for reinforcement learning.

    It can be used either over RL-Glue through the :class:`PinballRLGlue`
    adapter or interactively with :class:`PinballView`.

    """
    ACC_X = 0
    ACC_Y = 1
    DEC_X = 2
    DEC_Y = 3
    ACC_NONE = 4

    STEP_PENALTY = -1
    THRUST_PENALTY = -5
    END_EPISODE = 10000

    def __init__(self, configuration):
        """ Read a configuration file for Pinball and draw the domain to screen

        :param configuration: a configuration file containing the polygons,
            source(s) and target location.
        :type configuration: str

        """
        self.action_effects = {self.ACC_X:(1, 0), self.ACC_Y:(0, 1), self.DEC_X:(-1, 0), self.DEC_Y:(0, -1), self.ACC_NONE:(0, 0)}

        random.seed()

        # Set up the environment according to the configuration
        self.obstacles = []
        self.target_pos = []
        self.target_rad = 0.01

        ball_rad = 0.01
        start_pos = []
        with open(configuration) as fp:
            for line in fp.readlines():
                tokens = line.strip().split()
                if not len(tokens):
                    continue
                elif tokens[0] == 'polygon':
                    # legacy
                    # self.obstacles.append(
                    #     PinballObstacle(zip(*[iter(map(float, tokens[1:]))] * 2)))

                    # Parse coordinate pairs properly
                    coords = list(map(float, tokens[1:]))
                    if len(coords) % 2 != 0:
                        raise ValueError("Polygon must have even number of coordinates (x,y pairs)")

                    # Group coordinates into (x,y) pairs
                    points = []
                    for i in range(0, len(coords), 2):
                        points.append([coords[i], coords[i+1]])

                    # print("Creating obstacle with points:", points)
                    self.obstacles.append(PinballObstacle(points))
                elif tokens[0] == 'target':
                    self.target_pos = [float(tokens[1]), float(tokens[2])]
                    self.target_rad = float(tokens[3])
                elif tokens[0] == 'start':
                    coords = list(map(float, tokens[1:]))
                    if len(coords) % 2 != 0:
                        raise ValueError("Start positions must have even number of coordinates")

                    start_pos = []
                    for i in range(0, len(coords), 2):
                        start_pos.append([coords[i], coords[i+1]])

                    # print("Start positions:", start_pos)
                elif tokens[0] == 'ball':
                    ball_rad = float(tokens[1])

        self.ball = BallModel(list(random.choice(start_pos)), ball_rad)

    def get_state(self):
        """ Access the current 4-dimensional state vector

        :returns: a list containing the x position, y position, xdot, ydot
        :rtype: list

    	"""
        return [self.ball.position[0], self.ball.position[1], self.ball.xdot, self.ball.ydot]

    def take_action(self, action):
        """ Take a step in the environment

        :param action: The action to apply over the ball
            :type action: int

        """
        for i in range(20):
            if i == 0:
                self.ball.add_impulse(*self.action_effects[action])

            self.ball.step()

            # Detect collisions
            ncollision = 0
            dxdy = np.array([0, 0])

            for obs in self.obstacles:
                if obs.collision(self.ball):
                    dxdy = dxdy + obs.collision_effect(self.ball)
                    ncollision += 1

            if ncollision == 1:
                self.ball.xdot = dxdy[0]
                self.ball.ydot = dxdy[1]
                if i == 19:
                    self.ball.step()
            elif ncollision > 1:
                self.ball.xdot = -self.ball.xdot
                self.ball.ydot = -self.ball.ydot

            if self.episode_ended():
                return self.END_EPISODE

        self.ball.add_drag()
        self._check_bounds()

        if action == self.ACC_NONE:
            return self.STEP_PENALTY

        return self.THRUST_PENALTY

    def episode_ended(self):
        """ Find out if the ball reached the target

            :returns: True if the ball reched the target position
        :rtype: bool

    	"""
        return np.linalg.norm(np.array(self.ball.position)-np.array(self.target_pos)) < self.target_rad

    def _check_bounds(self):
        """ Make sure that the ball stays within the environment """
        if self.ball.position[0] > 1.0:
            self.ball.position[0] = 0.95
        if self.ball.position[0] < 0.0:
            self.ball.position[0] = 0.05
        if self.ball.position[1] > 1.0:
            self.ball.position[1] = 0.95
        if self.ball.position[1] < 0.0:
            self.ball.position[1] = 0.05

class PinballEnv(gym.Env):
    """ This class is a wrapper for the :class:`PinballModel` to
    make it compatible with the `gymnasium <https://gymnasium.farama.org/>`_ API.
    It allows the pinball domain to be used with RL algorithms
    that expect a gymnasium environment.
    """

    def __init__(self, configuration):
        """ Initialize the pinball environment

        :param configuration: The path to a configuration file for a :class:`PinballModel`
        :type configuration: str
        """
        super(PinballEnv, self).__init__()
        self.configuration = configuration
        self.pinball = None

        self._init_pinball()

    def _init_pinball(self):
        self.pinball = PinballModel(self.configuration)

        # Define the action and observation spaces
        self.action_space = gym.spaces.Discrete(5)  # Right (ACC_X), Up (ACC_Y), Left (DEC_X), Down (DEC_Y) and Do nothing(ACC_NONE)
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )  # 4-dimensional state: [x position, y position, xdot, ydot]

    def reset(self):
        """ Reset the environment to the initial state """
        self._init_pinball()
        return self.pinball.get_state(), {}

    def step(self, action):
        """ Take a step in the environment """
        reward = self.pinball.take_action(action)
        obs = self.pinball.get_state()
        done = self.pinball.episode_ended()
        return obs, reward, done, {}, {}

    def render(self, mode='human'):
        raise NotImplementedError("Rendering is not implemented in this environment. Use run_pinballview for rendering.")

    def close(self):
        """ Close the environment """
        if self.pinball is not None:
            self.pinball = None

class PinballView:
    """ This class displays a :class:`PinballModel`

    This class is used in conjunction with the :func:`run_pinballview`
    function, acting as a *controller*.

    We use `pygame <http://www.pygame.org/>` to draw the environment.

    """
    def __init__(self, screen, model):
        """
        :param screen: a pygame surface
        :type screen: :class:`pygame.Surface`
        :param model: an instance of a :class:`PinballModel`
        :type model: :class:`PinballModel`
        """
        self.screen = screen
        self.model = model

        self.DARK_GRAY = [64, 64, 64]
        self.LIGHT_GRAY = [232, 232, 232]
        self.BALL_COLOR = [0, 0, 255]
        self.TARGET_COLOR = [255, 0, 0]

        # Draw the background
        self.background_surface = pygame.Surface(screen.get_size())
        self.background_surface.fill(self.LIGHT_GRAY)
        for obs in model.obstacles:
            polygons = list( map(self._to_pixels, obs.points) )
            pygame.draw.polygon(self.background_surface, self.DARK_GRAY, polygons, 0)

        pygame.draw.circle(
            self.background_surface, self.TARGET_COLOR, self._to_pixels(self.model.target_pos), int(self.model.target_rad*self.screen.get_width()))

    def _to_pixels(self, pt):
        """ Converts from real units in the 0-1 range to pixel units

        :param pt: a point in real units
        :type pt: list
        :returns: the input point in pixel units
        :rtype: list

        """
        return [int(pt[0] * self.screen.get_width()), int(pt[1] * self.screen.get_height())]

    def blit(self):
        """ Blit the ball onto the background surface """
        self.screen.blit(self.background_surface, (0, 0))
        pygame.draw.circle(self.screen, self.BALL_COLOR,
                           self._to_pixels(self.model.ball.position), int(self.model.ball.radius*self.screen.get_width()))


def run_pinballview(width, height, configuration):
    """ Controller function for a :class:`PinballView`

    :param width: The desired screen width in pixels
    :type widht: int
    :param height: The desired screen height in pixels
    :type height: int
    :param configuration: The path to a configuration file for a :class:`PinballModel`
    :type configuration: str

    """
    # Launch interactive pygame
    pygame.init()
    pygame.display.set_caption('Pinball Domain')
    screen = pygame.display.set_mode([width, height])

    environment = PinballModel(configuration)
    environment_view = PinballView(screen, environment)

    actions = {pygame.K_RIGHT:PinballModel.ACC_X, pygame.K_UP:PinballModel.DEC_Y, pygame.K_LEFT:PinballModel.DEC_X, pygame.K_DOWN:PinballModel.ACC_Y}

    done = False
    clock = pygame.time.Clock()
    while not done:
        pygame.time.wait(50)

        user_action = PinballModel.ACC_NONE

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                user_action = actions.get(event.key, PinballModel.ACC_NONE)

        # Take action and see if episode ended
        if environment.take_action(user_action) == environment.END_EPISODE:
            done = True

        # Render current state
        environment_view.blit()
        pygame.display.flip()

        # Limit the frame rate to 60 FPS
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pinball domain')
    parser.add_argument('configuration', help='the configuration file')
    parser.add_argument('--width', action='store', type=int,
                        default=500, help='screen width (default: 500)')
    parser.add_argument('--height', action='store', type=int,
                        default=500, help='screen height (default: 500)')
    parser.add_argument('--gym', action='store_true',
                        help='use gymnasium environment instead of pinball view')
    args = parser.parse_args()

    if args.gym:
        # If gymnasium is requested, create the environment
        env = PinballEnv(args.configuration)
        state, info = env.reset()
        print("Pinball environment created with configuration:", args.configuration)
        print("Action space:", env.action_space)
        print("Observation space:", env.observation_space)
        print("Initial state:", state)
        print("You can now use this environment with RL algorithms.")
        # Note: You can add more code here to interact with the environment
        # using the gymnasium API, such as running episodes, training agents, etc.
    else:
        run_pinballview(args.width, args.height, args.configuration)
