cmake_minimum_required(VERSION 3.10)
# set(CMAKE_CXX_STANDARD 14)
set(projectname "perksCG.exe")
include_directories(${CMAKE_SOURCE_DIR}/common)
include_directories(${CMAKE_SOURCE_DIR}/matrix)
include_directories(${CMAKE_SOURCE_DIR}/util)
include_directories(${CMAKE_SOURCE_DIR}/cub)
include_directories(${CMAKE_SOURCE_DIR})
project(${projectname} CUDA CXX)
set (CMAKE_CXX_STANDARD 14)

# set(DEFAULT_COMPILER_FLAG "--std=c++14 -Xptxas \"-v -dlcm=cg\"")
set(DEFAULT_COMPILER_FLAG "--std=c++14 -Xptxas \"-v -dlcm=cg\"")
set(archlist " -gencode arch=compute_80,code=sm_80 ")
string(APPEND DEFAULT_COMPILER_FLAG "${archlist}")
#string(APPEND DEFAULT_COMPILER_FLAG "${archlist} -DENABLE_CPU_DEBUG_CODE")

# set_target_properties(${libraryname} PROPERTIES
                            # COMPILE_FLAGS " ${DEFAULT_COMPILER_FLAG} ")       
add_library("tridiag" ${CMAKE_SOURCE_DIR}/matrix/tridiag.cpp)
add_library("cg_cpu" ${CMAKE_SOURCE_DIR}/cg_cpu.cpp)
# set_target_properties
# add_executable(${projectname} ${CMAKE_SOURCE_DIR}/conjugateGradientMultiBlockCG.cu)
# set_target_properties(${projectname} PROPERTIES COMPILE_FLAGS "${DEFAULT_COMPILER_FLAG}")

#add_executable("coo" ${CMAKE_SOURCE_DIR}/conjugateGradientMultiBlockCG.cu )
#set_target_properties("coo" PROPERTIES COMPILE_FLAGS "${DEFAULT_COMPILER_FLAG} -DCOO")
#target_link_libraries("coo" PUBLIC "tridiag" "cg_cpu")
add_library("cg_baseline" ${CMAKE_SOURCE_DIR}/cg_driver.cu)
set_target_properties("cg_baseline" PROPERTIES COMPILE_FLAGS "${DEFAULT_COMPILER_FLAG} -DBASELINE,NOCOO")
#set_target_properties("cg_baseline" PROPERTIES COMPILE_FLAGS "${DEFAULT_COMPILER_FLAG} -DBASELINE")

add_library("timer"  ${CMAKE_SOURCE_DIR}/util/timer.cu)
target_link_libraries("cg_baseline" PUBLIC "tridiag" "cg_cpu" "cub_utils" "timer")

add_library("cub_utils"  ${CMAKE_SOURCE_DIR}/util/cub_utils.cu)

#add_executable("baseline" ${CMAKE_SOURCE_DIR}/conjugateGradientMultiBlockCG.cu )
add_executable("baseline" ${CMAKE_SOURCE_DIR}/main.cu )
# set_target_properties("baseline" PROPERTIES COMPILE_FLAGS "${DEFAULT_COMPILER_FLAG} -DBASELINE")
target_link_libraries("baseline" PUBLIC "cg_baseline" )