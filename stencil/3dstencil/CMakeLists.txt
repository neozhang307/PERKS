cmake_minimum_required(VERSION 3.10)
# cmake_minimum_required(VERSION 3.16)
# set the project name and version
project(PERKS_2DSTENCIL VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)



add_subdirectory(j3d7pt)
add_subdirectory(j3d13pt)
add_subdirectory(j3d17pt)
add_subdirectory(j3d27pt)
add_subdirectory(poisson)