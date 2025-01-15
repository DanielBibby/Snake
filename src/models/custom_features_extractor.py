from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from torch import nn


class CustomFlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space: The observation space of the environment
    """

    def __init__(self, observation_space) -> None:
        super().__init__(observation_space, sum(observation_space.nvec - 1))
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Observations arrive in shape (batch_size, 400) that is to say, already flattened.
        I reshape these and then flatten again, dropping the first vector to implement
        drop one hot encoding to account for linearity in feautres.

        NNs can handle linearity
        by shrinking weights of redundant features. But by engineering this I hope to improve training
        efficiency by removing any training cost associated with learning this linearity.
        """
        batch_size = observations.shape[0]
        reshaped_observations = observations.reshape(batch_size, 100, 4)

        return self.flatten(reshaped_observations[:, :, 1:])
