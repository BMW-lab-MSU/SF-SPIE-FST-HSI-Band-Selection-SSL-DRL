import gym
import numpy as np
from gym import spaces
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class BandSelectionEnv(gym.Env):
    def __init__(self, features, labels, max_bands=20):
        super(BandSelectionEnv, self).__init__()
        self.features = features
        self.labels = labels
        self.num_bands = features.shape[1]
        self.max_bands = max_bands
        self.state = np.zeros(self.num_bands, dtype=np.int32)
        self.action_space = spaces.Discrete(self.num_bands)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_bands,), dtype=np.int32)
        self.selected = set()
        self.done = False

    def reset(self):
        self.state = np.zeros(self.num_bands, dtype=np.int32)
        self.selected = set()
        self.done = False
        return self.state

    def step(self, action):
        # Mark selected band
        self.state[action] = 1
        self.selected.add(action)

        # Compute incremental reward
        reward = self._evaluate_subset()

        # End episode if max_bands reached
        if len(self.selected) >= self.max_bands:
            self.done = True

        return self.state, reward, self.done, {}

    def _evaluate_subset(self):
        # If no bands selected, reward=0
        if not self.selected:
            return 0.0
        
        # Train classifier on current subset
        selected_features = self.features[:, list(self.selected)]
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(selected_features, self.labels)
        preds = clf.predict(selected_features)
        acc = accuracy_score(self.labels, preds)

        # Penalize larger subsets slightly
        penalty = 0.01 * len(self.selected)
        return acc - penalty
