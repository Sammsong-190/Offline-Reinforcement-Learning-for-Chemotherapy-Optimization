"""
Entry point: Generate offline dataset
"""
from env.robust import set_seed
from data.generate import generate_dataset, save_dataset

if __name__ == '__main__':
    set_seed(42)  # train patients
    data = generate_dataset(n_trajectories=500)  # ~1e5 transitions for Offline RL
    save_dataset(data)
