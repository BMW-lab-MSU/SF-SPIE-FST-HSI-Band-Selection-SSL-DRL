import numpy as np
import os
import matplotlib.pyplot as plt
from env_band_selection import BandSelectionEnv
from agent_dqn import DQNAgent

# Paths
FEATURES_PATH = "/home/n51x164/SPIE-2025/BurnSSL-DRL/features/features.npy"
LABELS_PATH = "/home/n51x164/SPIE-2025/BurnSSL-DRL/patches1/patch_labels.npy"
OUTPUT_DIR = "/home/n51x164/SPIE-2025/BurnSSL-DRL/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RUNS = 5         # Number of DRL runs for stability
EPISODES = 50    # Episodes per run
MAX_BANDS = 165  # Total bands

# Load features & labels
features = np.load(FEATURES_PATH).reshape(-1, 32)
labels = np.load(LABELS_PATH)
if labels.shape[0] > features.shape[0]:
    labels = labels[:features.shape[0]]

num_bands = features.shape[1]

# Store results across runs
all_run_scores = np.zeros((RUNS, num_bands))
all_top_bands = []

for run in range(RUNS):
    print(f"\n🔥 === DRL Run {run+1}/{RUNS} ===")

    # Initialize environment & agent
    env = BandSelectionEnv(features, labels, max_bands=num_bands)
    agent = DQNAgent(state_size=env.num_bands, action_size=env.action_space.n)
    agent.epsilon = 1.0           # Start fully exploratory
    agent.epsilon_decay = 0.98    # Slow decay to encourage exploration

    band_rewards = np.zeros(num_bands)
    band_counts = np.zeros(num_bands)

    for ep in range(EPISODES):
        state = env.reset()
        total_reward = 0
        selected_bands = set()

        for _ in range(env.num_bands):
            action = agent.act(state)
            next_state, base_reward, done, _ = env.step(action)

            # ✅ Diversity-aware reward
            diversity_bonus = len(set(selected_bands | {action})) / num_bands
            reward = base_reward + 0.02 * diversity_bonus  # encourage variety
            
            selected_bands.add(action)
            band_rewards[action] += reward
            band_counts[action] += 1

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

        agent.replay()
        print(f"Episode {ep+1}/{EPISODES}, Total Reward: {total_reward:.4f}")

    # Compute scores for this run
    band_scores = band_rewards / (band_counts + 1e-6)
    all_run_scores[run] = band_scores

    sorted_scores = np.column_stack((np.arange(num_bands), band_scores))
    sorted_scores = sorted_scores[sorted_scores[:, 1].argsort()[::-1]]
    top_50_bands = sorted_scores[:50, 0].astype(int)
    all_top_bands.append(top_50_bands)

    print("\n=== Top 50 Bands (Run {}) ===".format(run+1))
    for idx, score in sorted_scores[:50]:
        print(f"Band {int(idx)}: Score = {score:.4f}")

# === Aggregate results across runs ===
avg_band_scores = np.mean(all_run_scores, axis=0)
sorted_avg_scores = np.column_stack((np.arange(num_bands), avg_band_scores))
sorted_avg_scores = sorted_avg_scores[sorted_avg_scores[:, 1].argsort()[::-1]]

# Consensus Top-50 Bands
consensus_top_50 = sorted_avg_scores[:50, 0].astype(int)

print("\n✅ === Consensus Top 50 Bands Across Runs ===")
for idx, score in sorted_avg_scores[:50]:
    print(f"Band {int(idx)}: Avg Score = {score:.4f}")

# Save outputs
np.save(os.path.join(OUTPUT_DIR, "band_scores_drl_multi.npy"), sorted_avg_scores)
np.save(os.path.join(OUTPUT_DIR, "consensus_top_bands_drl.npy"), consensus_top_50)

# === Wavelength Mapping ===
start_nm, end_nm = 400, 1000
wavelengths = np.linspace(start_nm, end_nm, num_bands)  # Map each band index to wavelength

# === Plot Band Scores with Wavelengths ===
plt.figure(figsize=(14, 6))
plt.bar(wavelengths, avg_band_scores, width=2, color='skyblue', label="Avg Band Score")
plt.scatter(wavelengths[consensus_top_50], avg_band_scores[consensus_top_50],
            color='red', label="Consensus Top-50", zorder=3)

# Annotate Top-10 wavelengths
for i in range(10):
    idx = int(consensus_top_50[i])
    plt.text(wavelengths[idx], avg_band_scores[idx]+0.002,
             f"{int(wavelengths[idx])}nm", ha='center', fontsize=8, color='red')

plt.xlabel("Wavelength (nm)", fontsize=14)
plt.ylabel("Avg DRL Reward Score", fontsize=14)
plt.title("DRL Band Importance (Mapped to Wavelengths, Multi-run Avg)", fontsize=16)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "band_scores_drl_wavelength.png"), dpi=300)
plt.show()

