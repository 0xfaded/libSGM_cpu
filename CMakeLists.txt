project(libsgm-cpu)

cmake_minimum_required(VERSION 3.15)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall -Wextra -pedantic -Wno-unused-parameter -Werror -Wfatal-errors)

add_subdirectory(src)
