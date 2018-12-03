#! /bin/bash

mkdir -p bin

gcc src/embeddingMatching.c -g -o bin/embeddingMatching -lm -pthread -Ofast -Wall -funroll-loops

