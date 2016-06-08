# Asyncronous RL in Tensorflow/Keras

![](http://g.recordit.co/BeiqC9l70B.gif)

## Intro
This is a Tensorflow/Keras implementation of asyncronous 1-step Q learning as described in ["Asynchronous Methods for Deep Reinforcement Learning"](http://arxiv.org/pdf/1602.01783v1.pdf).

Since we're using multiple actor-learner threads to stabilize learning in place of experience replay (which is super memory intensive), this runs comfortably on a macbook w/ 4g of ram.

It uses Keras to define the deep q network (see model.py), OpenAI's gym library to interact with the Atari Learning Environment (see atari_environment.py), and Tensorflow for optimization/execution (see async_dqn.py).

## Usage
###Training
To kick off training, run:
```python async_dqn.py --experiment breakout --game "Breakout-v0" --num_concurrent 8
```
Here we're organizing the outputs for the current experiment under a folder called 'breakout', choosing "Breakout-v0" as our gym environment, and running 8 actor-learner threads concurrently.

###Visualizing training with tensorboard
We collect episode reward stats and max q values that can be vizualized with tensorboard by running the following:
```
tensorboard --logdir /tmp/summaries/breakout
```
This is what my per-episode reward and average max q value curves looked like over the training period:


###Evaluation
To run an gym evaluation just turn the testing flag to True and hand in a current checkpoint file:
```python async_dqn.py --experiment breakout --testing True --checkpoint_path /tmp/breakout.ckpt-920000 --num_eval_episodes 100
```
After completing the eval, we can upload our eval file to OpenAI's site as follows:

###Next Steps
See a3c.py for a WIP async advantage actor critic implementation.

## Resources
I found these super helpful as general background materials for deep RL:

* [David Silver's "Deep Reinforcement Learning" lecture](http://videolectures.net/rldm2015_silver_reinforcement_learning/)
* [Nervana's Demystifying Deep Reinforcement Learning blog post](http://www.nervanasys.com/demystifying-deep-reinforcement-learning/)
