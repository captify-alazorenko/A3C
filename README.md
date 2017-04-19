## A3C (Based on Denny Britz implementation)

#### Requirements:
ffmpeg, requirements.txt

#### Running

```
./train.py --model_dir /tmp/a3c --env Breakout-v0 --t_max 5 --eval_every 300 --parallelism 8
```

See `./train.py --help` for a full list of options. Then, monitor training progress in Tensorboard:

```
tensorboard --logdir=/home/andriy/Code/PyCharmProjects/A3C/resources/results
```

#### Components

- [`train.py`](../../../../../PycharmProjects/A3C/train.py) contains the main method to start training.
- [`estimators.py`](../../../../../PycharmProjects/A3C/estimators.py) contains the Tensorflow graph definitions for the Policy and Value networks.
- [`worker.py`](../../../../../PycharmProjects/A3C/worker.py) contains code that runs in each worker threads.
- [`policy_monitor.py`](../../../../../PycharmProjects/A3C/policy_monitor.py) contains code that evaluates the policy network by running an episode and saving rewards to Tensorboard.
