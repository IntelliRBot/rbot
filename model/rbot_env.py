import gym
from gym import spaces

# gym env for RBot
# integrates with TI SensorTag for getting data

N_DISCRETE_ACTIONS = 2

PITCH_DATA_THRESHOLD = 45
ANGULAR_VELOCITY_THRESHOLD = 10
LINEAR_VELOCITY_THRESHOLD = 10

class RBotEnv(gym.Env):
    """
    Description:
        A self-balancing robot controlled by RPi, using a pair of wheels.
        The robot attempts to balance by either moving forward or backward only.
        The goal is to keep it upright, in the same way as OpenAI's CartPoleEnv
        This Gym env is referenced from CartPole 
        (https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py)

    Observation:
        Type: Box(4)
        Num     Observation
        0       pitch_data
        1       angular_velocity
        2       linear_velocity  

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Move RBot forward
        1     Move RBot backward

    Reward:
        Reward is 1 for every step taken, including the termination step.
        We follow CartPole's reward for the time being.

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        pitch_data is more than 45 degrees in either direction (RBot has fallen over)
        Episode length is greater than 200.

    Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials. (Following CartPole)
    """

    # The different modes for RBot
    metadata = {'render.modes': ['human']}

    def __init__(self, arg1, arg2, ...):
        super(RBotEnv, self).__init__()

        # Values to keep track
        self.pitch_data = 0
        self.angular_velocity = 0
        self.linear_velocity = 0
        self.state = (self.pitch_data, self.angular_velocity, self.linear_velocity)

        # Maxmimum values of all observation space values
        self.pitch_data_threshold = 45
        self.angular_velocity_threshold = 10
        self.linear_velocity_threshold = 10

        # Define action and observation space
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

        # Set the maximum values of each item in the observation space
        # state = [pitch_data, angular_velocity, linear_velocity]
        high = np.array([self.pitch_data_threshold, self.angular_velocity_threshold, 
                        self.linear_velocity_threshold], dtype=np.float32)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    # Execute one time step within the environment
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        # TODO put in the relevant function calls, using placeholder function calls for now
        # Sends the relevant action to RBot
        # send_action_to_rbot(action)

        # Fetch the relevant data from the SensorTag
        # self.pitch_data = SensorTag.get_pitch_data()
        # self.angular_velocity = SensorTag.get_angular_velocity()
        # self.linear_velocity = SensorTag.get_linear_velocity()

        # Check against Episode Termination, which is pitch_data exceeding 45 degrees
        done = bool(
            pitch_data < -self.pitch_data_threshold
            or pitch_data > self.pitch_data_threshold
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # RBot has fallen over
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    # Reset the state of the environment to an initial state
    def reset(self):
        self.pitch_data = 0
        self.angular_velocity = 0
        self.linear_velocity = 0
        
        self.steps_beyond_done = None
        return np.array(self.state)

    # "Render" the environment by printing the current state
    def render(self, mode='human'):
        print("Pitch: {}, Angular Velocity: {}, Linear Velocity: {}".format(self.pitch_data, self.angular_velocity, self.linear_velocity))
