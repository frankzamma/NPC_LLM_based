import pandas as pd

def salva_csv(metric_per_episode, metric_name, path):
    df = pd.DataFrame({
            "Episode": list(range(1, len(metric_per_episode) + 1)),
            metric_name: metric_per_episode
        })
    df.to_csv(path, index=False)