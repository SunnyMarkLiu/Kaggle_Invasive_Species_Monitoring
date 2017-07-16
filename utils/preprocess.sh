#!/usr/bin/env bash

echo '========== move train data =========='
python move_train_data.py
echo '=========== resize images ==========='
python resize_images.py
