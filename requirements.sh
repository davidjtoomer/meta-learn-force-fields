#!/bin/bash
pip freeze | grep -v "@ file" > requirements.txt
conda env export > environment.yml