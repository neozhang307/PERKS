
#set_directory_properties(PROPERTIES TYPE "js2d5pt")
#set_property(DIRECTORY ./ PROPERTY TYPE js2d5pt)

string(LENGTH "${CMAKE_SOURCE_DIR}" beg )
string(LENGTH "${CMAKE_CURRENT_SOURCE_DIR}" end)
math(EXPR beg "${beg} + 1" OUTPUT_FORMAT DECIMAL) 

string(SUBSTRING "${CMAKE_CURRENT_SOURCE_DIR}" ${beg} ${end} type)
#message("${beg} ${end} ${type}")

set(reference_goal "${type}_reference.exe")

add_definitions(-D${type})
add_library(${type}_jacobi_gold ./jacobi_gold.cpp)
add_library(${type}_jacobi_gold_gen ./jacobi_gold_gen.cpp)

#SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
#project(${reference_goal} CXX CUDA)
project(${reference_goal} CUDA)
add_executable(${reference_goal} ${CMAKE_SOURCE_DIR}/jacobi.driver.cpp)
target_link_libraries(${reference_goal} PUBLIC 
                            ${type}_jacobi_gold ${type}_jacobi_gold_gen)
set_target_properties(${reference_goal} PROPERTIES
                            COMPILE_FLAGS " -DREFCHECK ")       

#project(${naive_goal} CUDA)
set(archlist " -gencode arch=compute_80,code=sm_80 ")
string(APPEND archlist " -gencode arch=compute_70,code=sm_70 ")

set(DEFAULT_COMPILER_FLAG "-D_TIMER_,MULTIGEN,CHECK -DWARMUPRUN -Xptxas \"-v -dlcm=cg\"")
string(APPEND DEFAULT_COMPILER_FLAG "${archlist}")

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
    #add_library(${libraryname} ${CMAKE_SOURCE_DIR}/jacobi-star.cu)
    set_target_properties(${libraryname} PROPERTIES COMPILE_FLAGS "${local_compiler_flag}")

    add_executable(${local_goal} ${LOCAL_DRIVER})
    #add_executable(${local_goal} ${CMAKE_SOURCE_DIR}/jacobi.driver.cpp)
    target_link_libraries(${local_goal} PUBLIC ${libraryname} ${LOCAL_LIB} ${prefix}_jacobi_gold ${prefix}_jacobi_gold_gen)
endfunction()
#find_library("jacobi_naive")
function(mylib sublib subkernelfile compiler_flag)
    add_library("${sublib}" "${subkernelfile}")
    set_target_properties("${sublib}" PROPERTIES COMPILE_FLAGS "${compiler_flag}")
endfunction()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# mylib("jacobi_naive" "${CMAKE_SOURCE_DIR}/jacobi-naive.cu" "${DEFAULT_COMPILER_FLAG}")
# mylib("jacobi_baseline" "${CMAKE_SOURCE_DIR}/jacobi-baseline.cu" "${DEFAULT_COMPILER_FLAG}")
# mylib("jacobi_persistent" "${CMAKE_SOURCE_DIR}/jacobi-baseline.cu" "${DEFAULT_COMPILER_FLAG} -DPERSISTENT")
# mylib("jacobi_gen" "${CMAKE_SOURCE_DIR}/jacobi-general.cu" "${DEFAULT_COMPILER_FLAG}")

# subcompile(${type} ${DEFAULT_COMPILER_FLAG} "NAIVE" SOURCE "${CMAKE_SOURCE_DIR}/jacobi-2d.cu"  DRIVER "${CMAKE_SOURCE_DIR}/jacobi.driver.cpp" LIB "jacobi_naive")
# # subcompile(${type} ${DEFAULT_COMPILER_FLAG} "BASELINE_CM" SOURCE "${CMAKE_SOURCE_DIR}/jacobi-2d.cu" DRIVER "${CMAKE_SOURCE_DIR}/jacobi.driver.cpp")
# subcompile(${type} ${DEFAULT_COMPILER_FLAG} "BASELINE" SOURCE "${CMAKE_SOURCE_DIR}/jacobi-2d.cu" DRIVER "${CMAKE_SOURCE_DIR}/jacobi.driver.cpp" LIB "jacobi_baseline")
# subcompile(${type} ${DEFAULT_COMPILER_FLAG} "PERSISTENT" SOURCE "${CMAKE_SOURCE_DIR}/jacobi-2d.cu" DRIVER "${CMAKE_SOURCE_DIR}/jacobi.driver.cpp" LIB "jacobi_persistent")
# subcompile(${type} ${DEFAULT_COMPILER_FLAG} "GEN" SOURCE "${CMAKE_SOURCE_DIR}/jacobi-2d.cu" DRIVER "${CMAKE_SOURCE_DIR}/jacobi.driver.cpp" LIB "jacobi_gen")

# mylib("${type}_naive" "${CMAKE_SOURCE_DIR}/jacobi-naive.cu" "${DEFAULT_COMPILER_FLAG}")
# mylib("${type}_baseline" "${CMAKE_SOURCE_DIR}/jacobi-baseline.cu" "${DEFAULT_COMPILER_FLAG}")
# mylib("${type}_persistent" "${CMAKE_SOURCE_DIR}/jacobi-baseline.cu" "${DEFAULT_COMPILER_FLAG} -DPERSISTENT")
mylib("${type}_gen" "${CMAKE_SOURCE_DIR}/jacobi-general.cu" "${DEFAULT_COMPILER_FLAG}")

# subcompile(${type} ${DEFAULT_COMPILER_FLAG} "NAIVE" SOURCE "${CMAKE_SOURCE_DIR}/jacobi-2d.cu"  DRIVER "${CMAKE_SOURCE_DIR}/jacobi.driver.cpp" LIB "${type}_naive")
# # subcompile(${type} ${DEFAULT_COMPILER_FLAG} "BASELINE_CM" SOURCE "${CMAKE_SOURCE_DIR}/jacobi-2d.cu" DRIVER "${CMAKE_SOURCE_DIR}/jacobi.driver.cpp")
# subcompile(${type} ${DEFAULT_COMPILER_FLAG} "BASELINE" SOURCE "${CMAKE_SOURCE_DIR}/jacobi-2d.cu" DRIVER "${CMAKE_SOURCE_DIR}/jacobi.driver.cpp" LIB "${type}_baseline")
# subcompile(${type} ${DEFAULT_COMPILER_FLAG} "PERSISTENT" SOURCE "${CMAKE_SOURCE_DIR}/jacobi-2d.cu" DRIVER "${CMAKE_SOURCE_DIR}/jacobi.driver.cpp" LIB "${type}_persistent")
subcompile(${type} ${DEFAULT_COMPILER_FLAG} "GEN" SOURCE "${CMAKE_SOURCE_DIR}/jacobi-2d.cu" DRIVER "${CMAKE_SOURCE_DIR}/jacobi.driver.cpp" LIB "${type}_gen")
# subcompile(${type} ${DEFAULT_COMPILER_FLAG} "GEN" SOURCE "${CMAKE_SOURCE_DIR}/jacobi-2d.cu" DRIVER "${CMAKE_SOURCE_DIR}/jacobi.driver.cpp")


#subcompile(type DEFAULT_COMPILER_FLAG "BASELINE" SHAREDLIB ${type}_jacobi_gold ${type}_jacobi_gold_gen)
#subcompile(${type} DEFAULT_COMPILER_FLAG "NAIVE" SHAREDLIB ${type}_jacobi_gold ${type}_jacobi_gold_gen )
#set(naive_goal "${type}_naive.exe")
#set(NAIVE_COMPILER_FLAG "-DNAIVE ")
#string(APPEND NAIVE_COMPILER_FLAG "${DEFAULT_COMPIcudaLER_FLAG}")

#add_library(${type}_jacobi_naive ${CMAKE_SOURCE_DIR}/jacobi-star.cu)
#set_target_properties(${type}_jacobi_naive PROPERTIES
#                            COMPILE_FLAGS "${NAIVE_COMPILER_FLAG}")       

#add_executable(${naive_goal} ${CMAKE_SOURCE_DIR}/jacobi.driver.cpp)
#target_link_libraries(${naive_goal} PUBLIC 
#                ${type}_jacobi_naive ${type}_jacobi_gold ${type}_jacobi_gold_gen)