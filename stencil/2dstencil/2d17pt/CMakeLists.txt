
#set_directory_properties(PROPERTIES TYPE "js2d5pt")
#set_property(DIRECTORY ./ PROPERTY TYPE js2d5pt)
SET(CUDA_SEPARABLE_COMPILATION ON)
# find_package(CUDA QUIET REQUIRED)
find_package(CUDA REQUIRED)
# target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

string(LENGTH "${CMAKE_SOURCE_DIR}" beg )
string(LENGTH "${CMAKE_CURRENT_SOURCE_DIR}" end)
math(EXPR beg "${beg} + 1" OUTPUT_FORMAT DECIMAL) 

string(SUBSTRING "${CMAKE_CURRENT_SOURCE_DIR}" ${beg} ${end} type)
#message("${beg} ${end} ${type}")

set(reference_goal "${type}_reference.exe")
# set_properties(CUDA_ARCHITECTURES "70;80")

# add_definitions(-D${type})
add_library(${type}_jacobi_gold ./jacobi_gold.cpp)
add_library(${type}_jacobi_gold_gen ./jacobi_gold_gen.cpp)

#SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
#project(${reference_goal} CXX CUDA)
project(${reference_goal} LANGUAGES CUDA CXX)
add_executable(${reference_goal} ${CMAKE_SOURCE_DIR}/jacobi.driver.cpp)
target_link_libraries(${reference_goal} PUBLIC 
                            ${type}_jacobi_gold ${type}_jacobi_gold_gen)
set_target_properties(${reference_goal} PROPERTIES
                            COMPILE_FLAGS " -DREFCHECK ")       
# set_target_properties("${reference_goal}" PROPERTIES CUDA_ARCHITECTURES "70;80")

#project(${naive_goal} CUDA)
set(archlist " -gencode arch=compute_80,code=sm_80 ")
string(APPEND archlist " -gencode arch=compute_70,code=sm_70 ")

# set(DEFAULT_COMPILER_FLAG "-D_TIMER_,MULTIGEN,__PRINT__  -Xptxas \"-v -dlcm=cg\"")
# @LINGQI: previous experiments show that setting Xptxas might harm performance
# set(DEFAULT_COMPILER_FLAG "-D_TIMER_,MULTIGEN,__PRINT__  ")
set(DEFAULT_COMPILER_FLAG "-D_TIMER_ ")
# set(COMPILECG " -rdc=true -c ")
set(COMPILECG " -rdc=true ")

# set(DEFAULT_COMPILER_FLAG "-D_TIMER_,MULTIGEN,CHECK -DWARMUPRUN -Xptxas \"-v -dlcm=cg\"")
# string(APPEND DEFAULT_COMPILER_FLAG "${archlist}")

# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo --use_fast_math -rdc=true -lcudadevrt")


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

    message("${LOCAL_SOURCE}")
    message("${LOCAL_DRIVER}")
    message("${LOCAL_LIB}")

    add_library(${libraryname} ${LOCAL_SOURCE})
    set_target_properties(${libraryname} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set_target_properties(${libraryname} PROPERTIES COMPILE_FLAGS "${local_compiler_flag} ")
    set_target_properties(${libraryname} PROPERTIES CUDA_ARCHITECTURES "70;80")
    target_link_libraries(${libraryname} PUBLIC ${LOCAL_LIB} ${prefix}_jacobi_gold ${prefix}_jacobi_gold_gen ${CUDA_LIBRARIES})

    #add_library(${libraryname} ${CMAKE_SOURCE_DIR}/jacobi-star.cu)

    add_executable(${local_goal} ${LOCAL_DRIVER})
    # message("CHECK is ${CUDA_LIBRARIES}")
    set_target_properties(${local_goal}
                      PROPERTIES LANGUAGES CUDA LINKER_LANGUAGE CUDA)
    set_target_properties("${local_goal}" PROPERTIES CUDA_ARCHITECTURES "70;80")
    set_target_properties(${local_goal} PROPERTIES COMPILE_FLAGS "${local_compiler_flag}")
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

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
# include_directories("/usr/local/cuda/include")



mylib("${type}_naive" "${CMAKE_SOURCE_DIR}/jacobi-naive.cu" "${DEFAULT_COMPILER_FLAG}")
set_target_properties("${type}_naive" PROPERTIES CUDA_ARCHITECTURES "70;80")

mylib("${type}_naive_nvcc" "${CMAKE_SOURCE_DIR}/jacobi-naive.cu" "${DEFAULT_COMPILER_FLAG} -DNAIVENVCC")
set_target_properties("${type}_naive_nvcc" PROPERTIES CUDA_ARCHITECTURES "70;80")

mylib("${type}_baseline" "${CMAKE_SOURCE_DIR}/jacobi-baseline.cu" "${DEFAULT_COMPILER_FLAG} ")
set_target_properties("${type}_baseline" PROPERTIES CUDA_ARCHITECTURES "70;80")
# # mylib("${type}_baseline_async" "${CMAKE_SOURCE_DIR}/jacobi-baseline.cu" "${DEFAULT_COMPILER_FLAG}  -DSMASYNC")
# # set_target_properties("${type}_baseline_async" PROPERTIES CUDA_ARCHITECTURES "80")

# mylib("${type}_persistent" "${CMAKE_SOURCE_DIR}/jacobi-baseline.cu" "${DEFAULT_COMPILER_FLAG} ${COMPILECG}  -DPERSISTENT")
# set_target_properties("${type}_persistent" PROPERTIES CUDA_ARCHITECTURES "70;80")
# # mylib("${type}_persistent_async" "${CMAKE_SOURCE_DIR}/jacobi-baseline.cu" "${DEFAULT_COMPILER_FLAG} ${COMPILECG}  -DPERSISTENT -DSMASYNC")
# # set_target_properties("${type}_persistent_async" PROPERTIES CUDA_ARCHITECTURES "70;80")

mylib("${type}_gen" "${CMAKE_SOURCE_DIR}/jacobi-general.cu" "${DEFAULT_COMPILER_FLAG} ${COMPILECG} -DGEN ")
set_target_properties("${type}_gen" PROPERTIES CUDA_ARCHITECTURES "70;80")

# mylib("${type}_gen_async" "${CMAKE_SOURCE_DIR}/jacobi-general.cu" "${DEFAULT_COMPILER_FLAG} ${COMPILECG} -DGEN -DSMASYNC")
# set_target_properties("${type}_gen_async" PROPERTIES CUDA_ARCHITECTURES "70;80")


mylib("${type}_genwr" "${CMAKE_SOURCE_DIR}/jacobi-general-wrapper.cu" "${DEFAULT_COMPILER_FLAG} ${COMPILECG} ")
set_target_properties("${type}_genwr" PROPERTIES CUDA_ARCHITECTURES "70;80")

# mylib("${type}_genwr_small" "${CMAKE_SOURCE_DIR}/jacobi-general-wrapper.cu" "${DEFAULT_COMPILER_FLAG} ${COMPILECG} -DSMALL ")
# set_target_properties("${type}_genwr_small" PROPERTIES CUDA_ARCHITECTURES "70;80")


subcompile(${type} "${DEFAULT_COMPILER_FLAG} -D__PRINT__" "NAIVE" SOURCE "${CMAKE_SOURCE_DIR}/jacobi-2d.cu"  DRIVER "${CMAKE_SOURCE_DIR}/jacobi.driver.cpp" LIB "${type}_naive" )
subcompile(${type} "${DEFAULT_COMPILER_FLAG} -D__PRINT__" "NAIVENVCC" SOURCE "${CMAKE_SOURCE_DIR}/jacobi-2d.cu"  DRIVER "${CMAKE_SOURCE_DIR}/jacobi.driver.cpp" LIB "${type}_naive_nvcc" )

subcompile(${type} "${DEFAULT_COMPILER_FLAG} -D__PRINT__" "BASELINE" SOURCE "${CMAKE_SOURCE_DIR}/jacobi-2d.cu"  DRIVER "${CMAKE_SOURCE_DIR}/jacobi.driver.cpp" LIB "${type}_baseline" )
# subcompile(${type} "${DEFAULT_COMPILER_FLAG} -D__PRINT__" "BASELINE_CM" SOURCE "${CMAKE_SOURCE_DIR}/jacobi-2d.cu"  DRIVER "${CMAKE_SOURCE_DIR}/jacobi.driver.cpp" LIB "${type}_baseline" )
# subcompile(${type} "${DEFAULT_COMPILER_FLAG} -D__PRINT__" "PERSISTENT" SOURCE "${CMAKE_SOURCE_DIR}/jacobi-2d.cu"  DRIVER "${CMAKE_SOURCE_DIR}/jacobi.driver.cpp" LIB "${type}_persistent" )
subcompile(${type} "${DEFAULT_COMPILER_FLAG} -D__PRINT__" "GEN" SOURCE "${CMAKE_SOURCE_DIR}/jacobi-2d.cu"  DRIVER "${CMAKE_SOURCE_DIR}/jacobi.driver.cpp" LIB "${type}_gen" )
subcompile(${type} "${DEFAULT_COMPILER_FLAG} -D__PRINT__" "GENWR" SOURCE "${CMAKE_SOURCE_DIR}/jacobi-2d.cu"  DRIVER "${CMAKE_SOURCE_DIR}/jacobi.driver.cpp" LIB "${type}_genwr")


