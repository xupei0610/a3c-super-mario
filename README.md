# A3C Implementation for Playing Super Mario Bros.

This is an A3C implementation to train the agent playing Super Mario Bros.
This program is mainly for the course CSci5611 Animation & Planning in Games at UMN.

## Introduction

This program employs A3C and a network with LSTM to train the agent playing Super Mario Bros.

The game of Super Mario Bros. has an annoying pitfall that adverses the training: continuous jump is not allowed when the jump key (button `A`) is hold. This feature could make the agent stuck due to that it trends to hold the jump key at the beginning of training. This feature also make the vanlia DQN, who uses only CNN, work quite badly.

It takes me about two days to finish the training using 8 workers. The log file is provided in `pre-trained/a3c_log` foler. The model is too large to upload to github. You can download it from [my Google drive](https://drive.google.com/open?id=1Zi_M--BCCXygsWHmQFrbqsB__Em8MSir).


## Usage

Run

    python a3c.py --mode train

to train a model from scratch.

Change the log folder, checkpoint save folder and number of workers in `config.py`.

This program uses distributed Tensorflow in order to improve the performance of multi-threads.

Python Package Dependencies:
+ Tensorflow
+ OpenCV

Software Required:
+ Fceux

This program uses PIPE to communicate with Fceux and thus **only** support **Unit/Linux** platforms.

To run the test, please run the commands

    python a3c.py --mode test

You may need to change `A3C_SAVE_FOLDER` in `config.py` to the corresponding directory.

## Demos

![Demo](./docs/demo.gif)


## Credits

The code for communication between Fceux and python program is modified from [ppaquette/gym-super-mario](https://github.com/ppaquette/gym-super-mario).

I remove the requirement of OpenAI gym and simplify the code largely to improve the running performance.


