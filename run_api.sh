#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ $(uname -m) == 'arm64' ]]; then
        python app.py --execution-provider coreml
    else
        python app.py
    fi
else
    if command -v nvidia-smi &> /dev/null; then
        python app.py --execution-provider cuda
    else
        python app.py
    fi
fi
