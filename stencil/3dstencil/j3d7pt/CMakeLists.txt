SET(CUDA_SEPARABLE_COMPILATION ON)
# find_package(CUDA QUIET REQUIRED)
find_package(CUDA REQUIRED)

string(LENGTH "${CMAKE_SOURCE_DIR}" beg )
string(LENGTH "${CMAKE_CURRENT_SOURCE_DIR}" end)
math(EXPR beg "${beg} + 1" OUTPUT_FORMAT DECIMAL) 

string(SUBSTRING "${CMAKE_CURRENT_SOURCE_DIR}" ${beg} ${end} type)
message("${beg} ${end} ${type}")

set(reference_goal "${type}_reference.exe")

add_definitions(-D${type})
add_library(${type}_jacobi_gold ./${type}_gold.cpp)
add_library(${type}_jacobi_gold_gen ./${type}_gold_gen.cpp)

# SET (CMAKE_C_COMPILER_WORKS 1)
# SET (CMAKE_CXX_COMPILER_WORKS 1)
# set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
project(${reference_goal} CUDA CXX)
# find_package(CUDA)
# set(CMAKE_CUDA_STANDARD 14)

add_executable(${reference_goal} ${CMAKE_SOURCE_DIR}/j3d.driver.cpp)
target_link_libraries(${reference_goal} PUBLIC 
                            ${type}_jacobi_gold ${type}_jacobi_gold_gen)
set_target_properties(${reference_goal} PROPERTIES
                            COMPILE_FLAGS " -DREFCHECK ")     
                            
set_target_properties("${reference_goal}" PROPERTIES CUDA_ARCHITECTURES "70;80")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(archlist " -gencode arch=compute_80,code=sm_80 ")
string(APPEND archlist " -gencode arch=compute_70,code=sm_70 ")

set(DEFAULT_COMPILER_FLAG "-D_TIMER_ ")
set(COMPILECG " -rdc=true ")

set(DEFAULT_COMPILER_FLAG "-D_TIMER_ -Xptxas \"-v -dlcm=cg\" ${archlist}")

function(subcompile prefix default_compiler_flag aim)
    message("aim is ${aim}")
    message("prefix is ${prefix}")
    message("default compiler flag is ${default_compiler_flag}")
    string(TOLOWER ${aim} loweraim)
    set(local_goal "${prefix}_${loweraim}.exe")
    set(libraryname "${prefix}_jacobi_${loweraim}")
    set(local_compiler_flag "-D${aim} ")
    string(APPEND local_compiler_flag "${default_compiler_flag}")
    message("Local Compiler Flag is ${local_compiler_flag}")
    message("Local new library is ${libraryname}")
    message("GOAL is ${local_goal}")
    
    #message("${SOURCE}")
    #set(prefix DEMO)
    #set(flags IS_ASCII IS_UNICODE)
    #set(singleValues TARGET)
    #set(multiValues SOURCES RES)

    #include(CMakeParseArguments)
    
    #set(local_prefix LOCAL)
    set(multiValues SOURCE DRIVER LIB)

    cmake_parse_arguments(PARSE_ARGV 2 "LOCAL"
                     ""
                     ""
                     "${multiValues}")

    message("souce is ${LOCAL_SOURCE}")
    message("driver is ${LOCAL_DRIVER}")
    message("lib is ${LOCAL_LIB}")

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
    # set_target_properties(${local_goal} PROPERTIES COMPILE_FLAGS "${local_compiler_flag}")
   # set_target_properties(${local_goal} PROPERTIES COMPILE_FLAGS "${local_compiler_flag}")
    #add_executable(${local_goal} ${CMAKE_SOURCE_DIR}/jacobi.driver.cpp)
    set_target_properties(${local_goal} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)


    target_link_libraries(${local_goal} PUBLIC ${libraryname} ${prefix}_jacobi_gold ${prefix}_jacobi_gold_gen)
endfunction()
#find_library("jacobi_naive")
function(mylib sublib subkernelfile compiler_flag)
    add_library("${sublib}" "${subkernelfile}")
    set_target_properties("${sublib}" PROPERTIES COMPILE_FLAGS "${compiler_flag}")
endfunction()


# mylib("${type}_naive" "${CMAKE_SOURCE_DIR}/j3d-naive.cu" "${DEFAULT_COMPILER_FLAG} ")
# set_target_properties("${type}_naive" PROPERTIES CUDA_ARCHITECTURES "70;80")

mylib("${type}_baseline" "${CMAKE_SOURCE_DIR}/j3d-baseline.cu" "${DEFAULT_COMPILER_FLAG}  ")
set_target_properties("${type}_baseline" PROPERTIES CUDA_ARCHITECTURES "70;80")

mylib("${type}_baseline_memwarp" "${CMAKE_SOURCE_DIR}/j3d-baseline-memwarp.cu" "${DEFAULT_COMPILER_FLAG}  ")
set_target_properties("${type}_baseline" PROPERTIES CUDA_ARCHITECTURES "70;80")

# # subcompile(${type} )
# subcompile(${type} "${DEFAULT_COMPILER_FLAG} " "NAIVE" SOURCE "${CMAKE_SOURCE_DIR}/j3d.cu" DRIVER "${CMAKE_SOURCE_DIR}/j3d.driver.cpp" LIB "${type}_naive" )
subcompile(${type} "${DEFAULT_COMPILER_FLAG} " "BASELINE" SOURCE "${CMAKE_SOURCE_DIR}/j3d.cu" DRIVER "${CMAKE_SOURCE_DIR}/j3d.driver.cpp" LIB "${type}_baseline" )
subcompile(${type} "${DEFAULT_COMPILER_FLAG} " "BASELINE_MEMWARP" SOURCE "${CMAKE_SOURCE_DIR}/j3d.cu" DRIVER "${CMAKE_SOURCE_DIR}/j3d.driver.cpp" LIB "${type}_baseline_memwarp" )



# add_library("${type}_naive" "${CMAKE_SOURCE_DIR}/j3d-naive.cu")
# set_target_properties("${type}_naive" PROPERTIES CUDA_ARCHITECTURES "70;80")
# set_target_properties("${type}_naive" PROPERTIES COMPILE_FLAGS "${DEFAULT_COMPILER_FLAG}")

# set(libraryname "${type}_jacobi_naive")
# add_library(${libraryname}  "${CMAKE_SOURCE_DIR}/j3d.cu")
# set_target_properties(${libraryname} PROPERTIES CUDA_ARCHITECTURES "70;80")
# set_target_properties(${libraryname} PROPERTIES COMPILE_FLAGS "${DEFAULT_COMPILER_FLAG} -DNAIVE")

# set(local_goal "${type}_naive.exe")
# add_executable(${local_goal} "${CMAKE_SOURCE_DIR}/j3d.driver.cpp")
# # set_target_properties("${local_goal}" PROPERTIES CUDA_ARCHITECTURES "70;80")
# target_link_libraries(${local_goal} PUBLIC ${libraryname} ${type}_jacobi_gold_gen ${type}_jacobi_gold ${type}_naive )

