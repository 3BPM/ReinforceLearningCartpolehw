import numpy as np
import matplotlib.pyplot as plt
from env import BalancingCartEnv
from dqn_agent import DQNAgent
import time
import os
from configs import HIDDEN_SIZE,PATH
from vis import plot_training_results


def test_agent(env, agent, num_episodes=5):
    print("\nTesting trained agent...")
    agent.epsilon = 0  # Pure exploitation

    for ep in range(num_episodes):
        state, _ = env.reset()
        terminated = truncated = False
        total_reward = 0

        while not (terminated or truncated):
            action = agent.act(state)
            continuous_action = agent.get_continuous_action(action)

            state, reward, terminated, truncated, _ = env.step(continuous_action)
            total_reward += reward
            env.render()

        print(f"Test Ep {ep+1}: Reward = {total_reward:.1f}, "
              f"Steps = {env.current_step}/{env.max_episode_steps}")

if __name__ == '__main__':
    # Training configuration
    train_render_mode = None  # Set to 'human' for visualization (slower)
    num_training_episodes = 100000
    target_update_freq = 10  # Update target network every 10 episodes
    save_freq = 100  # Save model every 100 episodes

    # Create environment and agent
    env = BalancingCartEnv(render_mode=train_render_mode)
    # agent = DQNAgent(env,
    #                 gamma=0.99,
    #                 lr=1e-4,
    #                 batch_size=64,
    #                 memory_size=100000,
    #                 epsilon_start=1.0,
    #                 epsilon_end=0.01,
    #                 epsilon_decay=0.995)
    agent = DQNAgent(env,
                    gamma=0.99,
                    lr=1e-4,
                    batch_size=64,
                    memory_size=100000,
                    epsilon_start=1.0,
                    epsilon_end=0.01,
                    epsilon_decay=0.995,
                    hidden_size=HIDDEN_SIZE)

    # Load previous model if exists
    model_path = PATH
    agent.load(model_path)
    needtrain = not os.path.exists(model_path)
    if needtrain:
        # Training phase
        print(f"Starting training for {num_training_episodes} episodes...")
        rewards = []
        losses = []
        lengths = []
        start_time = time.time()

        for episode in range(1, num_training_episodes + 1):
            state, _ = env.reset()
            terminated = truncated = False
            episode_reward = 0
            episode_loss = 0
            step_count = 0

            while not (terminated or truncated):
                # Select and execute action
                action = agent.act(state)
                continuous_action = agent.get_continuous_action(action)
                next_state, reward, terminated, truncated, _ = env.step(continuous_action)

                # Store experience
                agent.remember(state, action, reward, next_state, terminated)

                # Train
                loss = agent.replay()
                if loss is not None:
                    episode_loss += loss

                state = next_state
                episode_reward += reward
                step_count += 1

            # Post-episode updates
            agent.update_epsilon()

            if episode % target_update_freq == 0:
                agent.update_target_network()

            # Record stats
            avg_loss = episode_loss / step_count if step_count > 0 else 0
            rewards.append(episode_reward)
            losses.append(avg_loss)
            lengths.append(env.current_step)

            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(rewards[-10:])
                avg_loss = np.mean(losses[-10:])
                elapsed_time = time.time() - start_time
                print(f"Ep {episode}/{num_training_episodes} | "
                    f"Avg R: {avg_reward:.1f} | "
                    f"Avg L: {avg_loss:.4f} | "
                    f"ε: {agent.epsilon:.3f} | "
                    f"Time: {elapsed_time:.1f}s")

            # Save model periodically
            if episode % save_freq == 0:
                agent.save(model_path)
                print(f"Model saved at episode {episode}")

        # Save final model
        agent.save(model_path)

        # Plot results
        plot_training_results(rewards, losses, lengths)

    # Testing phase with visualization
    test_env = BalancingCartEnv(render_mode='human')
    test_agent(test_env, agent)
    test_env.close()