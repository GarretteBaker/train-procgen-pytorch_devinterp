#!/bin/bash
apt update
apt install sudo
sudo apt install tmux
sudo apt install chmod
sudo apt-get install qtbase5-dev
pip install procgen gym gym3
pip install git+https://github.com/GarretteBaker/procgen-tools-dev-interp.git
pip install wandb devinterp tdqm tensorboard
wandb login a39441ba061c141c5884e1cf68cddcba99cf2dc3
pip install git+https://github.com/UlisseMini/circrl.git
pip install git+https://github.com/GarretteBaker/procgenAISC_devinterp.git
pip install plotly