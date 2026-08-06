# Root CMakeLists.txt — thin wrapper that delegates to cpp/.
#
# This repository has no unified top-level build; each language module is
# built independently (see the root justfile). This file exists only so
# `cmake -S . -B build` from the repo root also works for the C++ module.
cmake_minimum_required(VERSION 3.20)
project(dev_repo_template LANGUAGES CXX)

add_subdirectory(cpp)
