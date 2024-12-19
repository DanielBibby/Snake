from stable_baselines3.dqn.policies import DQNPolicy

from .custom_features_extractor import CustomFlattenExtractor


class CustomDQNPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs)
        # Use the custom feature extractor
        self.features_extractor = CustomFlattenExtractor(
            self.observation_space,
        )
