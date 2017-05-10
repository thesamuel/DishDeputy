cmake_minimum_required(VERSION 3.6)
project(DishDeputy)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

set(SOURCE_FILES main.cpp)
add_executable(DishDeputy ${SOURCE_FILES})
target_link_libraries(DishDeputy ${OpenCV_LIBS})
