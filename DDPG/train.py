import numpy as np
import gym
from ddpg_agent import DDPGAgent
from gym.wrappers import RecordVideo
import os
import torch
import json
import matplotlib.pyplot as plt

def train_agent(
    max_episodes=500, 
    early_stop_threshold=-200,
    save_dir='./checkpoints',
    log_dir='./logs',
    resume=False
):
    """
    训练DDPG智能体
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 环境初始化
    try:
        env = gym.make('Pendulum-v1')
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化智能体
        agent = DDPGAgent(state_size, action_size, device)
        
        # 训练状态变量
        scores = []
        avg_scores = []
        best_avg_score = float('-inf')
        start_episode = 0
        patience = 20
        no_improve = 0
        training_info = {}
        
        # 恢复训练
        if resume and os.path.exists(f"{save_dir}/checkpoint.pt"):
            checkpoint = torch.load(f"{save_dir}/checkpoint.pt")
            agent.load_state_dict(checkpoint['agent_state'])
            scores = checkpoint['scores']
            avg_scores = checkpoint['avg_scores']
            start_episode = checkpoint['episode']
            print(f"Resuming training from episode {start_episode}")
            
        # 主训练循环
        for episode in range(start_episode, max_episodes):
            state, _ = env.reset()
            episode_reward = 0
            actor_losses = []
            critic_losses = []
            
            while True:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.memory.push(state, action, reward, next_state, done)
                actor_loss, critic_loss = agent.train()
                
                if actor_loss is not None:
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
                    
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # 更新学习率
            agent.actor_scheduler.step()
            agent.critic_scheduler.step()
            
            scores.append(episode_reward)
            
            # 计算统计信息
            if len(scores) >= 10:
                avg_score = np.mean(scores[-10:])
                avg_scores.append(avg_score)
                
                # 早停检查
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    no_improve = 0
                    # 保存最佳模型
                    torch.save(agent.state_dict(), f"{save_dir}/best_model.pt")
                else:
                    no_improve += 1
                
                # 记录训练信息
                training_info[episode] = {
                    'episode_reward': episode_reward,
                    'avg_score': avg_score,
                    'actor_loss': np.mean(actor_losses) if actor_losses else 0,
                    'critic_loss': np.mean(critic_losses) if critic_losses else 0,
                    'noise_scale': agent.noise_scale
                }
                
                # 保存检查点
                if episode % 50 == 0:
                    checkpoint = {
                        'agent_state': agent.state_dict(),
                        'scores': scores,
                        'avg_scores': avg_scores,
                        'episode': episode
                    }
                    torch.save(checkpoint, f"{save_dir}/checkpoint.pt")
                
                # 打印训练信息
                if episode % 10 == 0:
                    print(f"\nEpisode {episode}")
                    print(f"Average Score: {avg_score:.2f}")
                    print(f"Actor Loss: {np.mean(actor_losses):.4f}")
                    print(f"Critic Loss: {np.mean(critic_losses):.4f}")
                    print(f"Noise Scale: {agent.noise_scale:.4f}")
                    print("-" * 50)
                
                # 检查是否达到目标或早停
                if avg_score >= early_stop_threshold or no_improve >= patience:
                    print(f"Training finished at episode {episode}")
                    break
        
        # 保存训练数据和绘制图表
        np.savetxt(f"{log_dir}/scores.csv", scores, delimiter=',')
        np.savetxt(f"{log_dir}/avg_scores.csv", avg_scores, delimiter=',')
        
        with open(f"{log_dir}/training_info.json", 'w') as f:
            json.dump(training_info, f)
            
        plot_training_results(scores, avg_scores, log_dir)
        
        return agent, scores, avg_scores
        
    except Exception as e:
        print(f"Training error: {str(e)}")
        raise
        
    finally:
        env.close()

def plot_training_results(scores, avg_scores, log_dir):
    """绘制训练结果图表"""
    plt.figure(figsize=(10, 5))
    plt.plot(scores, label='Episode Reward')
    plt.plot(avg_scores, label='Average Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(f"{log_dir}/training_curve.png")
    plt.close()