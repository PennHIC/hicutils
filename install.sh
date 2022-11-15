#!/bin/bash
set -e

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e hicutils
