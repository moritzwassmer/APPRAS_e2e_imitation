#TODO: You have to change! path to gitlab root folder and carla installation folder
setx GITLAB_ROOT C:\Users\morit\OneDrive\UNI\Master\WS22\APP-RAS\Programming
setx CARLA_ROOT D:\CARLA\WindowsNoEditor


#AUTOMATICALLY
setx LEADERBOARD_ROOT %GITLAB_ROOT%\transfuser_repo\transfuser\leaderboard
setx SCENARIO_RUNNER_ROOT %GITLAB_ROOT%\transfuser_repo\transfuser\scenario_runner


setx PYTHONPATH %PYTHONPATH%;%CARLA_ROOT%\PythonAPI\carla;%CARLA_ROOT%\PythonAPI\carla\dist\carla-0.9.10-py3.7-win-amd64.egg;%LEADERBOARD_ROOT%;%SCENARIO_RUNNER_ROOT%
