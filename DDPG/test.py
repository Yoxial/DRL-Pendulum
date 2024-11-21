# test.py
import gym
import torch
import numpy as np
from ddpg_agent import DDPGAgent
from utils import record_video

def test_agent(
    model_path='./checkpoints/best_model.pt',
    num_episodes=10
):
    """测试已训练的智能体"""
    # 初始化环境
    env = gym.make('Pendulum-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化智能体并加载模型
    agent = DDPGAgent(state_size, action_size, device)
    checkpoint = torch.load(model_path)
    if isinstance(checkpoint, dict) and 'actor' in checkpoint:
        # 如果是完整的状态字典
        agent.load_state_dict(checkpoint)
    else:
        # 如果只保存了模型参数
        agent.actor.load_state_dict(checkpoint)
    
    test_scores = []
    try:
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # 明确设置training=False
                action = agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state
            
            test_scores.append(episode_reward)
            print(f"Episode {episode}: Reward = {episode_reward:.2f}")
        
        print("\n=== 测试结果 ===")
        print(f"平均奖励: {np.mean(test_scores):.2f}")
        print(f"最大奖励: {np.max(test_scores):.2f}")
        print(f"最小奖励: {np.min(test_scores):.2f}")
        print(f"标准差: {np.std(test_scores):.2f}")
        
        # 记录一个测试视频
        record_video(agent)
        
    finally:
        env.close()

if __name__ == "__main__":
    test_agent()