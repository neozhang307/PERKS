cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(PERKS_2DSTENCIL VERSION 1.0)



#add_executable(PERKS_2DSTENCIL PERKS_2DSTENCIL.cxx)

#configure_file(PERKSConfig.h.in PERKSConfig.h)


#target_include_directories(PERKS_2DSTENCIL PUBLIC
#                           "${PROJECT_BINARY_DIR}"
#                           )

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# enable_language(CUDA)
# add_library("jacobi_naive" SHARED ${CMAKE_SOURCE_DIR}/jacobi-naive.cu)

# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/temp.sh ${CMAKE_CURRENT_BINARY_DIR}/temp.sh COPYONLY)
# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/iteratedomain.sh ${CMAKE_CURRENT_BINARY_DIR}/iteratedomain.sh COPYONLY)
# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/iteratemindomain.sh ${CMAKE_CURRENT_BINARY_DIR}/iteratemindomain.sh COPYONLY)
# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/iteratebaseline_cm.sh ${CMAKE_CURRENT_BINARY_DIR}/iteratebaseline_cm.sh COPYONLY)
# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/iterategen.sh ${CMAKE_CURRENT_BINARY_DIR}/iterategen.sh COPYONLY)
# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/iterategenwr.sh ${CMAKE_CURRENT_BINARY_DIR}/iterategenwr.sh COPYONLY)

# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/iteratefiles.sh ${CMAKE_CURRENT_BINARY_DIR}/iteratefiles.sh COPYONLY)

# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ProcessDomaindata.py ${CMAKE_CURRENT_BINARY_DIR}/ProcessDomaindata.py COPYONLY)
# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ProcessDomaindata.py ${CMAKE_CURRENT_BINARY_DIR}/ProcessDomaindata.py COPYONLY)

add_subdirectory(2d5pt)
add_subdirectory(2ds9pt)
add_subdirectory(2d13pt)
add_subdirectory(2d17pt)
add_subdirectory(2d21pt)
add_subdirectory(2ds25pt)

add_subdirectory(2d9pt)
add_subdirectory(2d25pt)