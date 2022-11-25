# End2Endappras



## Description

This repository was created for the course Applications of Robotics and Autonomous Systems at TU Berlin WS22/23. 
The project aims to create a End-to-End model for autonomous driving that is developed in the simulator Carla. 
Specifically, the project aims to investigate and compare the performance of several different sensor-setups. 
The set-ups investigated are the [following](https://gitlab.com/jakowiren/end2endappras/-/issues/18):

- RGB
- Odometry/IMU + RGB
- Odometry/IMU + RGB + LiDAR
- Odometry/IMU + RGB + RADAR
- Odometry/IMU + RGB + LiDAR + RADAR


## Architecture

The Master Thesis by Jonas Dammen "End-to-end deep learning for autonomous driving" compares three different model architectures for multimodal end-to-end driving in Carla. The result was that a stacked spatio-temporal CNN performed better than a RNN and a simple CNN. However, he also notes that he expects the temporal model to outperform the spatiotemporal model if properly configured (more difficult).
"A necessity for the spatiotemporal model was to find the correct size of the dense layers. Except for that the architecture is similar to the spatial one, except for the stacking of feature extractors. After a good combination of dense layers was
found, then tweaking of hyper-parameters was applied to get the final model. This was not the case with the temporal architecture. This architecture was without doubt the hardest to train, and to get to achieve a validation loss below 0.1. This
doesn’t mean that the temporal architecture is less suited for the task, but the complexity of a recurrent neural network makes it more difficult to configure. If the correct architecture and training configuration was found during exploitation, then
a certain guess would be that it would achieve better results on the task compared to the Spatiotemporal model. This comes from the fact that an LSTM uses a forget gate that enables it to filter what kind of features it should pass forward from
earlier time-steps to the dense layers, while the spatiotemporal model provides all the feature from all the time steps to the dense layers."

It is often normal to use the Nvidia CNN architecture for image input. 

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage





## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
In progress

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.com/jakowiren/end2endappras.git
git branch -M main
git push -uf origin main
```

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***
