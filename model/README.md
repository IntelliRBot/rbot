## DDQN Implementation with

### Setup
```
sudo apt install python3.7 python3.7-dev python3-pip
python3.7 -m pip install virtualenv
python3.7 -m virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

### Models
- DDQN (dqqn.py)
- Quantized DDQN for CartPoleV1 on TFLite (dqqn_quantized_cartpole.py)
- Quantized DDQN for Gazebo on TFLite (dqqn_quantized_gazebo.py)
```
python <filename>
```

### References
- https://github.com/rlcode/reinforcement-learning
- https://github.com/rlcode/reinforcement-learning/pull/85
