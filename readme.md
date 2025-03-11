This repo is just saving template code for later use.

It runs a huggingface model locally and you can ask questions on your notes.
Can be ran on linux with python 3.10 with cuda version 12.6.

Make sure that nvcc and nvidia-smi commands are working.
And if used with docker, make sure docker2 and docker-nvidia works fine to ensure that containers can access the gpu cuda cores.

(Did not try on tensor cores as is)