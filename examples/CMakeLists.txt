cmake_minimum_required(VERSION 3.15)
project(YoloInterfaceExample)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

add_executable(detect_example
    main.cc
)

target_link_libraries(detect_example PRIVATE
    yolo_interface
    ${OpenCV_LIBS}
)