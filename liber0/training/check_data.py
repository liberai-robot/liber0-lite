import argparse
import json
from pathlib import Path

import torch
import yaml

from ..backbone import _configure_ieee_float32_matmul
from .data import RoboDojoDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    _configure_ieee_float32_matmul()
    inference = yaml.safe_load((args.checkpoint / 'config.yaml').read_text())['inference']
    stats = json.loads((args.checkpoint / 'dataset_stats.json').read_text())
    data = RoboDojoDataset(args.dataset, stats, inference, seed=42, training=False, color={})
    for episode in data.episodes:
        for path in (data.episode_paths(episode['episode_index'])[0], *data.episode_paths(episode['episode_index'])[1]):
            if not path.is_file():
                raise FileNotFoundError(path)
    indices = sorted({0, data.episodes[0]['length'] - 1, len(data) - 1})
    for index in indices:
        sample = data[index]
        if not all(torch.isfinite(sample[key]).all() for key in ('video', 'action', 'proprio')):
            raise ValueError(f'Non-finite sample at {index}.')
        print(json.dumps(dict(index=index, video=list(sample['video'].shape), action=list(sample['action'].shape),
                              padded_actions=int(sample['action_is_pad'].sum()))))
    print(json.dumps(dict(episodes=len(data.episodes), frames=len(data),
                          verified='file inventory and selected decoded samples, not every frame')))


if __name__ == '__main__':
    main()
