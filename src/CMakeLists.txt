cmake_minimum_required(VERSION 3.8)
project(project_src)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        lambda_obj.c
        makefile arr_utils.h)

add_executable(project_src ${SOURCE_FILES})
target_link_libraries(project_src m)