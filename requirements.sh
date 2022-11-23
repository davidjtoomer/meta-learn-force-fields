#!/bin/bash
pip freeze | grep -v "@ file" > requirements.txt
conda env export | grep -v "prefix:" > environment.yml