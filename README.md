# End2Endappras
[System-Design-Specifications](Final SDS - Hand in this one.docx.pdf)

[Presentation](Final presentation APPRAS V2.pptx.pdf)

## Description

This repository was created for the course Applications of Robotics and Autonomous Systems at TU Berlin WS22/23. 
The project aims to create a End-to-End model for autonomous driving that is developed in the simulator Carla. 
Specifically, the project aims to investigate and compare the performance of several different sensor-setups. 
Due to the challenging nature of the project, only two sensor-setups and models for End-to-End autonomous driving were created and compared: RGB and RGB + LiDAR. 

## System Architecture
In the following system architecture diagram the data engineering related components are highlighted in gray, the data science related components in green and the evaluation related components in blue.
![System architecture of project](architecture.png "System architecture")

## System specifications

**CARLA Simulator**

Used for data generation and interaction of our trained agent with the environment. For this experiment we have used CARLA 0.9.10.1. 

**Expert Driver**

The expert driver is a hardcoded agent that can perform “perfect” driving in the CARLA simulator and is thus used for generation of the train and test data sets. It is a rule-based expert algorithm used and Its performance is an upper bound for the learning-based agent. The autopilot has access to the complete state of the environment including vehicle and pedestrian locations and actions. 

**Data Generator**

Used to initialize the CARLA simulator environment (i.e. driving conditions such as other vehicles, pedestrians, …) and to instantiate the expert driver. It then runs the simulation and saves the data. We have used noise injected data in our 
training to improve our agents.

**Train/Test Data**

This data is the foundation for model training and evaluation. The database stores the sparse state emitted by the CARLA Simulator. The sparse state consists of all defined sensor data, vehicle control signals and vehicle state measurements. 

**Train/Test Set Preprocessor**

Used to perform preprocessing tasks on the data stored in the Train/Test Data database to prepare them for neural network usage.

**Train/Test Data Preprocessed**

Is a database that stores the preprocessed data to make preprocessed data reusable.

**Neural Network Architecture**

Defines the neural network architecture that will be trained.

**Model Trainer**

Trains the given model architecture according to defined schedules, hyperparameters etc.

**Agent**

Is the end-to-end network that represents the learned driving policy and interacts with the CARLA simulator.

**Validator**

Similarly to the Data Generator, it initializes a simulator environment and initiates the Agent built upon the trained model. Then it runs the simulation and captures the driving performance of the autonomous agent for all tracks which are to be tested.

**Evaluation Data**

Contains the performance measures of the autonomous agents for every tested track.

**Evaluator/ Analyzer**

Visualizes and analyzes the evaluation data regarding overall performance of the autonomous agents. From that we are going to extract our results for the experiments i.e. comparing the driving performance between the different sensor setups.


## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage





## Authors and acknowledgment
Team Members:  Jakob Wiren, Julian Klitzing,  Moritz Wassmer, Ege Atesalp, Can Kayser.
Special thanks to all the guidance and support provided by the DAI-Laboratory of the Technical University of Berlin - especially to our supervisor Philipp Grosenick.



## Project status
Finished, with recommendations for further work specified. 

***
