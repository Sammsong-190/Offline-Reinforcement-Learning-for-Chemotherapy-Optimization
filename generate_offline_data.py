"""
Entry point: Generate offline dataset
Uses reward_v2 + balanced mix (60% expert, 20% balanced, 10% aggressive, 10% conservative)
to improve action coverage and encourage tumor clearance.
"""
from env.robust import set_seed
from data.generate import generate_dataset, save_dataset

if __name__ == '__main__':
    set_seed(42)
    data = generate_dataset(
        n_trajectories=500,
        use_reward_v2=True,
        randomize_patient=True,
    )
    save_dataset(data, "offline_dataset.npz")
