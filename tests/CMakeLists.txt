cmake_minimum_required(VERSION 3.15)
project(YoloInterfaceTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Test executable
add_executable(yolo_test
    test_main.cc
)

target_include_directories(yolo_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(yolo_test PRIVATE
    yolo_interface
)

add_test(NAME YoloInterfaceTest COMMAND yolo_test)