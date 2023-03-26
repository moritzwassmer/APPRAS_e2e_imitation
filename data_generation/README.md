# Data Generation

## TransFuser Dataset

To be able to run the data generation scripts in the TransFuser Paper, we first write the correct environment variables in the [datagen script ](transfuser_repo/transfuser/leaderboard/scripts/datagen.sh) inside the TransFuser repo (change the CARLA_ROOT and WORK_DIR to your settings), then run the script. The environment dependencies are in the [requirements file](data_generation/requirements.txt).

```Shell
$ pip install -r requirements.txt
```

## Noise Injection

We have used noise injected data in our training to improve our agents. Since the existing dataset comes without any noise, we have used the TransFuser data generation script and a Carla Client running in a local computer to create our own noisy data. Noise is injected in intervals of 10 seconds (Cycles of 20 seconds, 10 seconds of noise followed by 10 second of recovery and data generation during recovery). The aforementioned script (and the requirements.txt for environment) is provided in the data_generation folder. To use it, please replace [autopilot.py](transfuser_repo/transfuser/team_code_autopilot/autopilot.py) file with [autopilot_noise.py](data_generation/autopilot_noise.py), and then run [datagen script ](transfuser_repo/transfuser/leaderboard/scripts/datagen.sh).