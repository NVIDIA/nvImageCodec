#!/bin/bash

# Different chroma subsampling factors
magick padlock-406986_640_420.jpg -sampling-factor 4:1:0 padlock-406986_640_410.jpg
magick padlock-406986_640_420.jpg -sampling-factor 4:1:1 padlock-406986_640_411.jpg
magick padlock-406986_640_420.jpg -sampling-factor 4:2:2 padlock-406986_640_422.jpg
magick padlock-406986_640_420.jpg -sampling-factor 4:4:0 padlock-406986_640_440.jpg
magick padlock-406986_640_420.jpg -sampling-factor 4:4:4 padlock-406986_640_444.jpg
