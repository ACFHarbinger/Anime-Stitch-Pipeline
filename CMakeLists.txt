# Root CMakeLists.txt — thin wrapper that delegates to base/.
#
# This repository has no unified top-level build; each module is built
# independently (see the root justfile). This file exists only so
# `cmake -S . -B build` from the repo root also works for the C++ module.
cmake_minimum_required(VERSION 3.20)
project(anime_stitch_pipeline LANGUAGES CXX)

add_subdirectory(base)
