set WORK_DIR=%GITLAB_ROOT%\MyTransfuser\transfuser

set SCENARIOS=%WORK_DIR%\leaderboard\data\training\scenarios\Scenario10\Town10HD_Scenario10.json
set ROUTES=%WORK_DIR%\leaderboard\data\training\routes\Scenario10\Town10HD_Scenario10.xml
set REPETITIONS=1
set CHALLENGE_TRACK_CODENAME=MAP
set CHECKPOINT_ENDPOINT=%WORK_DIR%\results\Town10HD_Scenario10.json
set SAVE_PATH=%GITLAB_ROOT%\data_generation\results
set TEAM_AGENT=%WORK_DIR%\team_code_autopilot\data_agent.py
set DEBUG_CHALLENGE=0
set RESUME=1
set DATAGEN=1

python %LEADERBOARD_ROOT%\leaderboard\leaderboard_evaluator_local.py ^
--scenarios=%SCENARIOS%  ^
--routes=%ROUTES% ^
--repetitions=%REPETITIONS% ^
--track=%CHALLENGE_TRACK_CODENAME% ^
--checkpoint=%CHECKPOINT_ENDPOINT% ^
--agent=%TEAM_AGENT% ^
--agent-config=%TEAM_CONFIG% ^
--debug=%DEBUG_CHALLENGE% ^
--resume=%RESUME%
