"""
Generate offline dataset v2: improved behavior policy + reward_v2
60% expert, 20% balanced, 10% aggressive, 10% conservative
"""
from env.robust import set_seed
from data.generate import generate_dataset_v2, save_dataset

if __name__ == "__main__":
    set_seed(42)
    data = generate_dataset_v2(
        n_trajectories=500,
        use_reward_v2=True,
        randomize_patient=True,
    )
    save_dataset(data, "offline_dataset.npz")
