#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python inference.py static/jason-calacanis.jpg static/ape-calacanis.png
CUDA_VISIBLE_DEVICES=1 python inference.py static/david-friedberg.png static/ape-friedberg.png
CUDA_VISIBLE_DEVICES=1 python inference.py static/chamath-palihapitiya.jpg static/ape-palihapitiya.png
CUDA_VISIBLE_DEVICES=1 python inference.py static/david-sacks.jpg static/ape-sacks.png