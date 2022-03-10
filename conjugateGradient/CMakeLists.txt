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
#set(DEFAULT_COMPILER_FLAG "--std=c++14")

set(DEFAULT_COMPILER_FLAG "--std=c++14 -Xptxas \"-v -dlcm=cg\"")
set(archlist " -gencode arch=compute_80,code=sm_80 ")
string(APPEND archlist " -gencode arch=compute_70,code=sm_70 ")

string(APPEND DEFAULT_COMPILER_FLAG "${archlist}")
string(APPEND DEFAULT_COMPILER_FLAG " -D__PRINT__ ")

add_library("tridiag" ${CMAKE_SOURCE_DIR}/matrix/tridiag.cpp)
add_library("cg_cpu" ${CMAKE_SOURCE_DIR}/cg_cpu.cpp)

add_library("timer"  ${CMAKE_SOURCE_DIR}/util/timer.cu)
add_library("cub_utils"  ${CMAKE_SOURCE_DIR}/util/cub_utils.cu)

# set_target_properties
# add_executable(${projectname} ${CMAKE_SOURCE_DIR}/conjugateGradientMultiBlockCG.cu)
# set_target_properties(${projectname} PROPERTIES COMPILE_FLAGS "${DEFAULT_COMPILER_FLAG}")

#add_executable("coo" ${CMAKE_SOURCE_DIR}/conjugateGradientMultiBlockCG.cu )
#set_target_properties("coo" PROPERTIES COMPILE_FLAGS "${DEFAULT_COMPILER_FLAG} -DCOO")
#target_link_libraries("coo" PUBLIC "tridiag" "cg_cpu")

# add_library("cg_baseline" ${CMAKE_SOURCE_DIR}/cg_driver.cu)
# set_target_properties("cg_baseline" PROPERTIES COMPILE_FLAGS "${DEFAULT_COMPILER_FLAG} -DBASELINE,NOCOO")
#set_target_properties("cg_baseline" PROPERTIES COMPILE_FLAGS "${DEFAULT_COMPILER_FLAG} -DBASELINE")



# add_library("cg_baseline" ${CMAKE_SOURCE_DIR}/cg_driver.cu)
# set_target_properties("cg_baseline" PROPERTIES COMPILE_FLAGS "${DEFAULT_COMPILER_FLAG} -DBASELINE,NOCOO")
# set_target_properties("cg_baseline" PROPERTIES CUDA_ARCHITECTURES "70;80")
# target_link_libraries("cg_baseline" PUBLIC "tridiag" "cg_cpu" "cub_utils" "timer")

# add_executable("baseline"  ${CMAKE_SOURCE_DIR}/main.cu)
# set_target_properties("baseline" PROPERTIES COMPILE_FLAGS "${DEFAULT_COMPILER_FLAG}")
# target_link_libraries("baseline" PUBLIC "cg_baseline" )


function(subcompile prefix compilerflag aim)
    message("aim is ${aim}")
    message("prefix is ${prefix}")
    message("default compiler flag is ${compilerflag}")
    string(TOLOWER ${aim} loweraim)
    set(local_goal "${prefix}_${loweraim}.exe")
    set(libraryname "${prefix}_${loweraim}")
    
    set(local_compiler_flag "${compilerflag}")
    message("Local Compiler Flag is ${local_compiler_flag}")

    message("Local Compiler Flag is ${compilerflag}")
    message("Local new library is ${libraryname}")
    message("GOAL is ${local_goal}")

    set(multiValues SOURCE DRIVER LIB)
    cmake_parse_arguments(PARSE_ARGV 2 "LOCAL"
                     ""
                     ""
                     "${multiValues}")
    message("Local source file is ${LOCAL_SOURCE}")
    message("Local driver file is ${LOCAL_DRIVER}")

    add_library(${libraryname} ${LOCAL_SOURCE})
    set_target_properties(${libraryname} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set_target_properties(${libraryname} PROPERTIES COMPILE_FLAGS "${local_compiler_flag} ")
    set_target_properties(${libraryname} PROPERTIES CUDA_ARCHITECTURES "70;80")
    target_link_libraries(${libraryname} PUBLIC ${LOCAL_LIB}  ${CUDA_LIBRARIES})

    #add_library(${libraryname} ${CMAKE_SOURCE_DIR}/jacobi-star.cu)

    add_executable(${local_goal} ${LOCAL_DRIVER})
    # message("CHECK is ${CUDA_LIBRARIES}")
    set_target_properties(${local_goal}
                      PROPERTIES LANGUAGES CUDA LINKER_LANGUAGE CUDA)
    set_target_properties("${local_goal}" PROPERTIES CUDA_ARCHITECTURES "70;80")
    set_target_properties(${local_goal} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set_target_properties(${local_goal} PROPERTIES COMPILE_FLAGS "${local_compiler_flag} ")

    target_link_libraries(${local_goal} PUBLIC ${libraryname})
endfunction()

subcompile("cg" "${DEFAULT_COMPILER_FLAG}  -DBASELINE,NOCOO,__PRINT__" "BASELINE" SOURCE "${CMAKE_SOURCE_DIR}/main.cu" DRIVER "${CMAKE_SOURCE_DIR}/cg_driver.cu" LIB "tridiag" "cg_cpu" "cub_utils" "timer" )


