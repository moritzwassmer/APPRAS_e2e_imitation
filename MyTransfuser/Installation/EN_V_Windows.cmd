
#TODO: You have to change!
setx GITLAB_ROOT C:\Users\egeat\Desktop\end2endappras 
setx CARLA_ROOT C:\Users\egeat\Desktop\CARLA_0.9.10.1\WindowsNoEditor


#AUTOMATICALLY
setx LEADERBOARD_ROOT %GITLAB_ROOT%\MyTransfuser\transfuser\leaderboard
setx SCENARIO_RUNNER_ROOT %GITLAB_ROOT%\MyTransfuser\transfuser\scenario_runner


setx PYTHONPATH %PYTHONPATH%;%CARLA_ROOT%\PythonAPI\carla;%CARLA_ROOT%\PythonAPI\carla\dist\carla-0.9.10-py3.7-win-amd64.egg;%LEADERBOARD_ROOT%;%SCENARIO_RUNNER_ROOT%
