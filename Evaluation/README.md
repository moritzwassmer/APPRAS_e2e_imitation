# Evaluation
## agents

The different agents in this folder are mostly examples copied from the origininal leaderboard repository which were usefull to test functionalities of the leaderboard. The Agents can be used by passing the file path to the argument `--agent` of the `leaderboard_evaluator.py`

The RGB and Lidar model also use `config.py` which provide configurations for the carla simulator like sensor configurations

## console_code
Examples for running the rgb and lidar agents on the longest6 Benchmark are provided.
It basically passes routes, scenarios, the agent and other arguments to the `leaderboard_evaluator.py` which configures carla to run the respective scenarios.

## results
This folder contains the longest6 Benchmark results of the rgb and lidar model. 
to_be_parsed stores the raw .json output of the `leaderboard_evaluator.py` which can then be processed by the `result_parser.py` or `pretty_print_json.py` to retrieve summarised statistics of the tested routes and scenarios or maps of where infractions occured.
Code to run the result_parser is provided in `result_parser.txt`.

## routes
Contains official routes of Carla and routes which were used to generate the trainingsdata as well as the routes used by the longest6benchmark


## scenarios
Contains official scenarios of Carla and routes which were used to generate the trainingsdata as well as the routes used by the longest6benchmark

### Longest6 benchmark

The Longest6 benchmark consists of 36 routes with an average route length of 1.5km, which is similar to the average route length of the official leaderboard (~1.7km). During evaluation, we ensure a high density of dynamic agents by spawning vehicles at every possible spawn point permitted by the CARLA simulator. Following the [NEAT evaluation benchmark](https://github.com/autonomousvision/neat/blob/main/leaderboard/data/evaluation_routes/eval_routes_weathers.xml), each route has a unique environmental condition obtained by combining one of 6 weather conditions (Cloudy, Wet, MidRain, WetCloudy, HardRain, SoftRain) with one of 6 daylight conditions (Night, Twilight, Dawn, Morning, Noon, Sunset).

More information can be found in the Transfuser Repository.


