[//]: # (Image References)



# Udacity Reinforcement Learning Nanodegree - Course "Value-Based Methods" Project : Navigation

# Udacity Reinforcement Learning Nanodegree - Course “Value-Based Methods” Project : Navigation

## Presentation
This project is the validation test for course 2 **Value-Based Methods** of Udacity's [Deep Reinforcement Learning - Nanodegree Program](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

It involves training an agent with a value-based method to solve the Unity [[#Unity “Banana”]] environment.

This project requires a special python development environment, the instructions for which are provided below.

## Description of the Unity environment "Banana"
> Provided by Udacity. These description is the one given in the project description section.

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Run this project

### 1. installation of the python runtime environment
**As of the date of availability** of this project version in Github, these instructions take up those available in the Udacity nanoprogram student knowledge base, in addition to [those provided](https://github.com/udacity/Value-based-methods) in the project description (and which present anomalies depending on the student's platform) :

**INSTALLATION INSTRUCTIONS FOR WINDOWS/LINUX/MACOS**
1. `conda create --name banana python=3.8 -c conda-forge`
2. `conda activate banana`
3. Create a directory where you want this save the project, say `banana_project`.
4. `cd banana_project`
5. `git clone https://github.com/udacity/Value-based-methods.git`
6. `cd Value-based-methods/python`
7. Open `requirements.txt` with a text editor and make the following changes:
	- **Remove all version numbers** behind each module except `protobuf`
	- Set the version of `protobuf` to **3.20.3**;
	- Remove the `torch` package; it will be installed later;
	- Save the changes and close the file.
8. `pip install .` in order to install almost all the required modules.
9. `pip install tqdm` also.
10. `python -m ipykernel install --user --name banana --display-name "banana"`
11. In order to install `PyTorch`, choose the right parameters in the [PyTorch installation page](https://pytorch.org/get-started/locally/), (here macos is the selected platform (*see figure above*)), and copy/paste/execute the suggested command, like `pip install torch torchvision torchaudio`.
![[screenshot_1440.png]]

### 2. Install this project
1. Go to a directory into which you want to isntall the project.
2. `git clone https://github.com/nicolasguillard/Udacity-RL-nanodegree-program---Course-2-Project.git`. The folder `Udacity-RL-nanodegree-program---Course-2-Project` is created.

#### File structure of this project
- `dqn_extensions/`: directory containing extension codes for the DQN algorithm.
	- `ext__exp_replay.py` : code of Experience Replay extension.
- `dqn_agent__expreplay_fixedqtarget.py`: code for the agent based on the DQN algorithm with Experience Replay and Fixed Q-Target.
- `model_weights_f275_solved.pth`: weights of the agent model trained via the `Training DQN Agent.ipynb` notebook. 
- `Navigation.ipynb` : notebook used to test access to the Unity Banana environment [[#3. Installation of the Unity “Banana” environment]] .
- `README.md` : this file.
- Report.ipynb`: project report including implementation description.
- `Training DQN Agent (fr).ipynb`: training and evaluation execution log, with explanations in French.
- `Training DQN Agent.ipynb`: training and evaluation logbook, with explanations in English.

### 3. Install Unity environment "Banana"
> Provided by Udacity. These description is the updated one given in the project description section.

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the repository, in the `Udacity-RL-nanodegree-program---Course-2-Project` folder, and unzip (or decompress) the file. 

## Run this project
Of course, a local Jupyter instance must be launched.

To verify the installation of the python runtime environment and access to the Unity “Banana” environment application, follow the instructions in `Navigation.ipynb`.

To run the agent training and validation project, follow the instructions in `Training DQN Agent.ipynb`.


