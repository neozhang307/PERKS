
#set_directory_properties(PROPERTIES TYPE "js2d5pt")
#set_property(DIRECTORY ./ PROPERTY TYPE js2d5pt)

string(LENGTH "${CMAKE_SOURCE_DIR}" beg )
string(LENGTH "${CMAKE_CURRENT_SOURCE_DIR}" end)
math(EXPR beg "${beg} + 1" OUTPUT_FORMAT DECIMAL) 

string(SUBSTRING "${CMAKE_CURRENT_SOURCE_DIR}" ${beg} ${end} type)
message("${beg} ${end} ${type}")

set(filename "${type}_reference.exe")

project(${filename})

add_library(${type}_jacobi_gold ./jacobi_gold.cpp)
add_library(${type}_jacobi_gold_gen ./jacobi_gold_gen.cpp)


add_executable(${filename} ../jacobi.driver.cpp)
target_link_libraries(${filename} PUBLIC ${type}_jacobi_gold ${type}_jacobi_gold_gen)
add_definitions(-D${type})