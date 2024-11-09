import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import imageio

from actor_critic import ActorCritic
from grid_env import GridEnvironment

# Function to render the environment's current state using Matplotlib
def render_episode(frames):
    plt.figure(figsize=(8, 8))
    for idx, frame in enumerate(frames):
        plt.imshow(frame)
        plt.axis('off')
        plt.title(f"Step: {idx + 1}")
        plt.show()
        plt.pause(0.1)  # Pause to create an animation effect
    plt.close()

# PPO loss function
def ppo_loss(model, old_logits, old_values, advantages, states, actions, returns, action_size, clip_ratio, optimizer, epochs):
    def compute_loss(logits, values, actions, returns):
        actions_onehot = tf.one_hot(actions, action_size, dtype=tf.float32)
        policy = tf.nn.softmax(logits)
        action_probs = tf.reduce_sum(actions_onehot * policy, axis=1)
        old_policy = tf.nn.softmax(old_logits)
        old_action_probs = tf.reduce_sum(actions_onehot * old_policy, axis=1)

        # Policy loss
        ratio = tf.exp(tf.math.log(action_probs + 1e-10) - tf.math.log(old_action_probs + 1e-10))
        clipped_ratio = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio)
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

        # Value loss
        value_loss = tf.reduce_mean(tf.square(values - returns))

        # Entropy bonus (optional)
        entropy_bonus = tf.reduce_mean(policy * tf.math.log(policy + 1e-10))

        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus  # Entropy regularization
        return total_loss

    def get_advantages(returns, values):
        advantages = returns - values
        return (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

    def train_step(states, actions, returns, old_logits, old_values):
        with tf.GradientTape() as tape:
            logits, values = model(states)
            loss = compute_loss(logits, values, actions, returns)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    advantages = get_advantages(returns, old_values)
    for _ in range(epochs):
        loss = train_step(states, actions, returns, old_logits, old_values)
    return loss

# Main function
def main(args):

    # Environment setup
    env = GridEnvironment(sx=0,sy=0,gy=9,gx=9, grid_size=10)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Hyperparameters
    gamma = 0.99
    lr_actor = 0.001
    clip_ratio = 0.2
    epochs = 10
    max_episodes = 1000000
    max_steps_per_episode = 1000000

    if args.mode == 'existing':
        # Load model
        try:
            model = tf.keras.models.load_model(args.input_model)
            print("Model loaded successfully.")
        except:
            model = ActorCritic(state_size, action_size)
            print("Model loaded failed.")

    else:
        # Initialize actor-critic model and optimizer
        model = ActorCritic(state_size, action_size)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)


        # Recompile the model if necessary
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        model.compile(optimizer=optimizer)

    # Main training loop
    for episode in range(max_episodes):
        states, actions, rewards, values, returns = [], [], [], [], []
        frames = []
        state = env.reset()
        for step in range(max_steps_per_episode):
            state = tf.expand_dims(tf.convert_to_tensor(state), 0)
            logits, value = model(state)

            # Sample action from the policy distribution
            action = tf.random.categorical(logits, 1)[0, 0].numpy()
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)


            frame = env.render(mode='rgb_array')
            frames.append(frame)

            state = next_state

            if done:
                returns_batch = []
                discounted_sum = 0
                for r in rewards[::-1]:
                    discounted_sum = r + gamma * discounted_sum
                    returns_batch.append(discounted_sum)
                returns_batch.reverse()

                states = tf.concat(states, axis=0)
                actions = np.array(actions, dtype=np.int32)
                values = tf.concat(values, axis=0)
                returns_batch = tf.convert_to_tensor(returns_batch)
                old_logits, _ = model(states)

                # Calculate Loss
                loss = ppo_loss(model, old_logits, values, returns_batch - np.array(values),
                                states, actions, returns_batch, action_size, clip_ratio, optimizer, epochs)

                # Get total reward
                total_reward = np.sum(rewards)

                print(f"Episode: {episode + 1}, Loss: {loss.numpy()}, Total Reward: {total_reward}")

                # Save the model and GIF
                if episode % 10 == 0:
                    model.save(os.path.join(f'{os.getcwd()}', 'ppo_model_checkpoints' ,f'ppo_model_episode_{episode + 1}.keras'))
                    gif_path = os.path.join(f'{os.getcwd()}', 'ppo_model_gifs', f'episode_{episode + 1}.gif')
                    imageio.mimsave(gif_path, frames, fps=10)
                    print(f"Model and GIF saved for Episode {episode + 1}")

                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO model training")

    # existing, new, or test model
    parser.add_argument('--mode', dest='mode', type=str, default='new', help='Enter new if you want a new model to be tested from scratch, '
                                                                'test if you want to test an existing model, or existing'
                                                                ' if you want to train existing model')
    parser.add_argument('--input_model', dest='input_model', type=str, default='', help='Path to existing model')

    args = parser.parse_args()
    main(args)

