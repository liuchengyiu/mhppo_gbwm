import numpy as np
import gym
TRAINING_EVALUATION_RATIO = 4
RUNS = 5
EPISODES_PER_RUN = 400
STEPS_PER_EPISODE = 200

if __name__ == "__main__":
    agent_results = []
    for run in range(RUNS):
        agent = SACAgent(env)
        run_results = []
        for episode_number in range(EPISODES_PER_RUN):
            print('\r', f'Run: {run + 1}/{RUNS} | Episode: {episode_number + 1}/{EPISODES_PER_RUN}', end=' ')
            evaluation_episode = episode_number % TRAINING_EVALUATION_RATIO == 0
            episode_reward = 0
            state = env.reset()
            done = False
            i = 0
            while not done and i < STEPS_PER_EPISODE:
                i += 1
                action = agent.get_next_action(state, evaluation_episode=evaluation_episode)
                next_state, reward, done, info = env.step(action)
                if not evaluation_episode:
                    agent.train_on_transition(state, action, next_state, reward, done)
                else:
                    episode_reward += reward
                state = next_state
            if evaluation_episode:
                run_results.append(episode_reward)
        agent_results.append(run_results)
