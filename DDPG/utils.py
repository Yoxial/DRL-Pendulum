import os
import gym
from gym.wrappers import RecordVideo
from typing import Optional
import datetime

def record_video(
    agent, 
    env_name: str = 'Pendulum-v1',
    video_dir: str = "./videos",
    max_steps: Optional[int] = None,
    video_name: Optional[str] = None
) -> float:
    try:
        # 生成唯一的视频文件名和文件夹
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_folder = os.path.join(video_dir, f"run_{timestamp}")
        
        # 如果文件夹已存在，先删除它以确保清洁的环境
        if os.path.exists(video_folder):
            shutil.rmtree(video_folder)
        
        # 创建新的文件夹
        os.makedirs(video_folder)
        
        # 设置视频名称
        video_name = video_name or f"episode_{timestamp}"
        
        # 创建环境
        env = gym.make(env_name, render_mode="rgb_array")
        env = RecordVideo(
            env, 
            video_folder,
            episode_trigger=lambda x: True,
            name_prefix=video_name
        )
        
        # 重置环境
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # 执行环境交互
        while not done:
            # 检查是否达到最大步数
            if max_steps and steps >= max_steps:
                break
                
            # 选择动作并执行
            action = agent.select_action(state, training=False)  # 测试模式
            state, reward, terminated, truncated, _ = env.step(action)
            
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
        return total_reward
        
    except Exception as e:
        print(f"录制视频时发生错误: {str(e)}")
        return 0.0
        
    finally:
        env.close()
