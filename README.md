This is a fork of the [MicroRTS-py](https://github.com/Farama-Foundation/MicroRTS-Py) repository.
All additional code is in the `mini/` directory.
This directory contains:

- `mini/agents/`: The different agents that can be used to play the game.
- `mini/eval.py`: Runs a few games between two agents and returns the results. It also displays the matches in a graphical interface.
- `mini/tournament.py`: Runs a tournament between all specified agents and displays the results.
- `mini/train.py`: Trains an agent.
- `mini/explore.py`: Trains multiple agents with different hyperparameters.
- `mini/parameters.py`: The parameters used by all the other files.
