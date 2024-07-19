# Part 1 - Importing Libraries and Defining Classes

# Importing necessary libraries
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import gymnasium as gym
from collections import deque, namedtuple
import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gym.wrappers.monitoring.video_recorder import VideoRecorder

# Define the neural network architecture
class Network(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(Network, self).__init__()
        # Set the random seed for reproducibility
        self.seed = torch.manual_seed(seed)
        # Define fully connected layers
        self.fully_connected_layer_1 = nn.Linear(state_size, 64)
        self.fully_connected_layer_2 = nn.Linear(64, 64)
        self.fully_connected_layer_3 = nn.Linear(64, action_size)

    def forward(self, state):
        # Define the forward pass through the network
        x = self.fully_connected_layer_1(state)
        x = F.relu(x)  # Apply ReLU activation function
        x = self.fully_connected_layer_2(x)
        x = F.relu(x)  # Apply ReLU activation function
        return self.fully_connected_layer_3(x)  # Output layer

# Define the replay memory
class ReplayMemory(object):
    def __init__(self, capacity):
        # Determine the device to use (GPU if available)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity  # Maximum size of the replay memory
        self.memory = []  # Initialize an empty list to store experiences

    def push(self, event):
        # Add a new experience to the memory
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            # Remove the oldest experience if memory exceeds capacity
            del self.memory[0]

    def sample(self, batch_size):
        # Sample a batch of experiences from the memory
        experiences = random.sample(self.memory, k=batch_size)
        # Convert sampled experiences to tensors
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return states, next_states, actions, rewards, dones
# Part 2 - Defining the DQN Agent and Initializing the Environment

# Define the DQN agent
class Agent():
    def __init__(self, state_size, action_size):
        # Determine the device to use (GPU if available)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size  # Size of the state space
        self.action_size = action_size  # Size of the action space
        # Create local and target Q-Networks
        self.local_Q_Network = Network(state_size, action_size).to(self.device)
        self.target_Q_Network = Network(state_size, action_size).to(self.device)
        # Define optimizer
        self.optimizer = optim.Adam(self.local_Q_Network.parameters(), lr=learning_rate)
        # Initialize replay memory
        self.memory = ReplayMemory(replay_buffer_size)
        self.time_step = 0  # Initialize time step counter

    def step(self, state, action, reward, next_state, done):
        # Store the experience in replay memory
        self.memory.push((state, action, reward, next_state, done))
        # Update time step
        self.time_step = (self.time_step + 1) % 4
        # Learn every 4 time steps
        if self.time_step == 0:
            if len(self.memory.memory) > min_batch_size:
                # Sample a batch of experiences from memory
                experiences = self.memory.sample(100)
                # Perform learning step
                self.learn(experiences, discount_factor)

    def act(self, state, epsilon=0.):
        # Select action using epsilon-greedy policy
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_Q_Network.eval()  # Set network to evaluation mode
        with torch.no_grad():
            # Get action values from local Q-Network
            action_values = self.local_Q_Network(state)
        self.local_Q_Network.train()  # Set network back to training mode
        if random.random() > epsilon:
            # Exploit: select action with highest value
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # Explore: select a random action
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, discount_factor):
        # Perform a learning step with a batch of experiences
        states, next_states, actions, rewards, dones = experiences
        # Get max predicted Q values for next states from target model
        next_q_target = self.target_Q_Network(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        q_target = rewards + (discount_factor * next_q_target * (1 - dones))
        # Get expected Q values from local model
        q_expected = self.local_Q_Network(states).gather(1, actions)
        # Compute loss
        loss = F.mse_loss(q_expected, q_target)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Update target network
        self.soft_update(self.local_Q_Network, self.target_Q_Network, interpolation_parameter)

    def soft_update(self, local_model, target_model, interpolation_parameter):
        # Soft update model parameters
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(interpolation_parameter * local_param.data + (1.0 - interpolation_parameter) * target_param.data)

# Function to display the video
def show_video():
    mp4list = glob.glob('*.mp4')  # Get list of .mp4 files
    if len(mp4list) > 0:
        mp4 = mp4list[0]  # Get the first video file
        video = io.open(mp4, 'r+b').read()  # Read video file
        encoded = base64.b64encode(video)  # Encode video in base64
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))  # Display video
    else:
        print("Could not find video")

# Function to record video of the trained agent
def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')  # Create environment with video rendering
    state, _ = env.reset()  # Reset the environment
    done = False
    frames = []  # List to store frames
    while not done:
        frame = env.render()  # Render the frame
        frames.append(frame)  # Append frame to list
        action = agent.act(state)  # Select an action
        state, reward, done, _, _ = env.step(action.item())  # Take a step
    env.close()  # Close the environment
    imageio.mimsave('video.mp4', frames, fps=30)  # Save frames as a video

# Initialize the environment
env = gym.make('LunarLander-v2')
# Get state and action space sizes
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_of_action = env.action_space.n
print("State Shape: " + str(state_shape) + "\n" +
      "State Size: " + str(state_size) + "\n" +
      "Number Of Actions: " + str(number_of_action))

# Set hyperparameters
learning_rate = 5e-4
min_batch_size = 100
discount_factor = 0.999
replay_buffer_size = int(1e7)
interpolation_parameter = 1e-3

# Initialize the agent
agent = Agent(state_size, number_of_action)
# Part 3 - Training the DQN Agent and Displaying the Video of the Trained Agent

# Training the agent
number_episodes = 2000  # Number of episodes to train
maximum_number_timesteps_per_episode = 1000  # Max timesteps per episode
epsilon_starting_value = 1.0  # Initial epsilon value for epsilon-greedy policy
epsilon_ending_value = 0.01  # Minimum epsilon value
epsilon_decay_value = 0.995  # Epsilon decay rate
epsilon = epsilon_starting_value  # Set initial epsilon
scores_on_100_episodes = deque(maxlen=100)  # To store scores of last 100 episodes

for episode in range(1, number_episodes + 1):
    state, _ = env.reset()  # Reset the environment
    score = 0  # Initialize score for the episode
    for t in range(maximum_number_timesteps_per_episode):
        action = agent.act(state, epsilon)  # Select an action
        next_state, reward, done, _, _ = env.step(action)  # Take a step in the environment
        agent.step(state, action, reward, next_state, done)  # Store experience and learn
        state = next_state  # Update state
        score += reward  # Accumulate reward
        if done:
            break  # End the episode if done
    scores_on_100_episodes.append(score)  # Save the score
    epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)  # Decay epsilon
    print("\rEpisode {}\tAverage Score: {:.2f}".format(episode, np.mean(scores_on_100_episodes)), end="")
    if episode % 100 == 0:
        # Print average score every 100 episodes
        print("\rEpisode {}\tAverage Score: {:.2f}".format(episode, np.mean(scores_on_100_episodes)))
    if np.mean(scores_on_100_episodes) >= 250.0:
        # Check if environment is solved
        print("\nEnvironment Solved in {:d} episodes! \tAverage Score: {:.2f}".format(episode - 100, np.mean(scores_on_100_episodes)))
        torch.save(agent.local_Q_Network.state_dict(), 'checkpoint.pth')  # Save model weights
        break  # Exit training loop


# Show video of the trained agent
show_video_of_model(agent, 'LunarLander-v2')


# Display the video
show_video()
