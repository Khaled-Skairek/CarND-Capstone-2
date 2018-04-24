#!/bin/bash -e


# clean
rm -rf devel
rm -rf build
catkin_make clean
rosclean

# run
catkin_make
