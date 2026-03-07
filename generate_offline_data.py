"""
Entry point: Generate offline dataset
Uses reward_v3 + balanced mix + state noise. n_trajectories=1000 for ~280k transitions.
"""
from env.robust import set_seed
from data.generate import generate_dataset, save_dataset

if __name__ == '__main__':
    set_seed(42)
    data = generate_dataset(
        n_trajectories=1000,
        use_reward_v3=True,
        randomize_patient=True,
        state_noise_sigma=0.02,
        expert_balance_ratio=0.6,
    )
    save_dataset(data, "offline_dataset.npz")
