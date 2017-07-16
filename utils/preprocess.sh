#!/usr/bin/env bash

echo '========== move train data =========='
python move_train_data.py
echo '========== preprocess images =========='
python image_preprocess.py
