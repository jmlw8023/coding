cmake_minimum_required(VERSION 3.15)
project(YoloDemo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# For demo, use local library files
set(LOCAL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(LOCAL_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")

# Find OpenCV (optional, for more advanced usage)
find_package(OpenCV QUIET COMPONENTS core imgproc)

# Create demo executable
add_executable(yolo_demo main.cpp)

# Include directories
target_include_directories(yolo_demo PRIVATE ${LOCAL_INCLUDE_DIR})

# Link libraries
if(OpenCV_FOUND)
    target_link_libraries(yolo_demo PRIVATE ${OpenCV_LIBS})
endif()

target_link_libraries(yolo_demo PRIVATE
    "${LOCAL_LIB_DIR}/yolo_interface.lib"
)

# Set output directory
set_target_properties(yolo_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# If using DLL, copy it to output directory
add_custom_command(TARGET yolo_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${LOCAL_LIB_DIR}/yolo_interface.dll"
        "${CMAKE_BINARY_DIR}/bin/"
)

message(STATUS "Demo configured successfully")
message(STATUS "OpenCV: ${OpenCV_FOUND}")