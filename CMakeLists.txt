cmake_minimum_required(VERSION 3.15)
project(YoloInterface VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(YOLO_WITH_OPENCV "Enable OpenCV backend" OFF)
option(YOLO_WITH_OPENVINO "Enable OpenVINO backend" OFF)
option(YOLO_WITH_TENSORRT "Enable TensorRT backend" OFF)
option(YOLO_BUILD_EXAMPLES "Build examples" OFF)

# Compiler flags for Windows
if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra)
endif()

# Source files - always include CPU backend
set(YOLO_SOURCES
    src/yolo_factory.cc
    src/backend/cpu_backend.cc
)

# Add OpenCV backend if explicitly enabled and available
if(YOLO_WITH_OPENCV)
    find_package(OpenCV QUIET COMPONENTS core imgproc dnn)
    if(OpenCV_FOUND)
        list(APPEND YOLO_SOURCES src/backend/opencv_backend.cc)
        add_compile_definitions(YOLO_OPENCV_BACKEND=1)
        message(STATUS "OpenCV backend: enabled (${OpenCV_VERSION})")
    else()
        message(WARNING "OpenCV not found, disabling OpenCV backend")
    endif()
endif()

# Add OpenVINO backend if explicitly enabled and found
if(YOLO_WITH_OPENVINO)
    find_package(OpenVINO QUIET)
    if(OpenVINO_FOUND)
        list(APPEND YOLO_SOURCES src/backend/openvino_backend.cc)
        add_compile_definitions(YOLO_OPENVINO_BACKEND=1)
        message(STATUS "OpenVINO backend: enabled")
    else()
        message(WARNING "OpenVINO not found, disabling OpenVINO backend")
    endif()
endif()

# Create shared library
add_library(yolo_interface SHARED ${YOLO_SOURCES})

# Add define for DLL exports
target_compile_definitions(yolo_interface PRIVATE YOLO_EXPORTS)

target_include_directories(yolo_interface PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(OpenCV_FOUND)
    target_link_libraries(yolo_interface PUBLIC ${OpenCV_LIBS})
endif()

# Set output name and location
set_target_properties(yolo_interface PROPERTIES
    OUTPUT_NAME "yolo_interface"
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Install targets
install(TARGETS yolo_interface
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

if(YOLO_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()