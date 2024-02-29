import torch
from env import String_Generator
from Q_network import TransformerQNetwork
from agent import DQNAgent
import copy
import datetime

if __name__== '__main__':
    input_size = 5
    hidden_size = 128
    num_layers = 2
    num_heads = 4
    output_size = 2  #logits of adding zero or one
    num_episodes = 5000
    done = False
    total_step = 0
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    env = String_Generator(stochastic_length=False, fixed_length = input_size)
    agent = DQNAgent(input_size, output_size, hidden_dim=64, env = env)
    
    for episode in range(num_episodes):
        start_state = env.reset()
        state = copy.deepcopy(start_state)

        for step in range(input_size):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn()
            agent.store_exp(state, action, next_state, reward, done, env.goal)
            state = next_state
            total_step += 1

        agent.update_q_network()
        accuracy = env.return_accuracy()
        agent.writer.add_scalar("accuracy", accuracy, episode + 1)
        print(f"Episode {episode + 1}, accuracy: {accuracy}")

    # save model
    torch.save(agent.q_network.state_dict(), f'epoch{num_episodes}_length{input_size}_{current_time}.pth')
    agent.writer.close()