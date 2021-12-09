string(LENGTH "${CMAKE_SOURCE_DIR}" beg )
string(LENGTH "${CMAKE_CURRENT_SOURCE_DIR}" end)
math(EXPR beg "${beg} + 1" OUTPUT_FORMAT DECIMAL) 

string(SUBSTRING "${CMAKE_CURRENT_SOURCE_DIR}" ${beg} ${end} type)
message("${beg} ${end} ${type}")

set(reference_goal "${type}_reference.exe")

add_definitions(-D${type})
add_library(${type}_gold ./${type}_gold.cpp)
add_library(${type}_gold_gen ./${type}_gold_gen.cpp)

SET (CMAKE_C_COMPILER_WORKS 1)
SET (CMAKE_CXX_COMPILER_WORKS 1)
# set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
project(${reference_goal} CUDA CXX)
# find_package(CUDA)
# set(CMAKE_CUDA_STANDARD 14)

add_executable(${reference_goal} ${CMAKE_SOURCE_DIR}/j3d.driver.cpp)
target_link_libraries(${reference_goal} PUBLIC 
                            ${type}_gold ${type}_gold_gen)
set_target_properties(${reference_goal} PROPERTIES
                            COMPILE_FLAGS " -DREFCHECK ")     
                            
set_target_properties("${reference_goal}" PROPERTIES CUDA_ARCHITECTURES "70;80")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})