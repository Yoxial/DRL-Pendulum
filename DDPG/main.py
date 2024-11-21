from train import train_agent
from utils import record_video

if __name__ == "__main__":
    trained_agent, scores, avg_scores = train_agent()
    # 记录最终训练效果视频
    record_video(trained_agent)
