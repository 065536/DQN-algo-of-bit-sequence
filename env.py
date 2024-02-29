import copy
import random
import gym

class String_Generator(gym.Env):

    def __init__(self, stochastic_length = False, fixed_length = 0):
        self.stochastic_length = stochastic_length
        if self.stochastic_length:
            self.goal_length = random.randint(1, 50)
        else:
            self.goal_length = fixed_length
    
    def generate_string(self):
        self.goal = [random.choice([0, 1]) for _ in range(self.goal_length)]
        return self.goal

    def reset(self):
        if self.stochastic_length:
            self.goal_length = random.randint(1, 50)
        self.goal = self.generate_string()
        self.step_count = 0
        self.state = [2 for _ in range(self.goal_length)]
        state = self.state
        self.next_state = copy.deepcopy(self.state)
        self.done = False
        return state

    def step(self, action):
        assert action == 0 or action == 1
        self.next_state[self.step_count] = action
        self.step_count += 1
        self.reward = self.get_reward()
        self.done = self.check_done()
        self.state = self.next_state
        return self.state, self.reward, self.done, {}
    
    def get_reward(self):
        assert self.goal_length >= self.step_count
        if self.next_state[self.step_count - 1] == self.goal[self.step_count - 1]:
            return 1
        else:
            return 0
    
    def check_done(self):
        if self.step_count >= self.goal_length:
            return True
        else:
            return False
        
    def return_accuracy(self):
        assert self.goal_length == self.step_count
        equal_count = sum(1 for state, goal in zip(self.state, self.goal) if state == goal)
        similarity_ratio = equal_count / self.goal_length
        return similarity_ratio


            


