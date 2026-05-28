cmake_minimum_required(VERSION 3.15)
project(YoloInterface VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================
# 构建选项
# ============================================================
option(YOLO_WITH_OPENCV "启用 OpenCV 后端" OFF)
option(YOLO_WITH_OPENVINO "启用 OpenVINO 后端" OFF)
option(YOLO_WITH_TENSORRT "启用 TensorRT 后端" OFF)
option(YOLO_BUILD_EXAMPLES "编译示例程序" OFF)
option(YOLO_BUILD_TESTS "编译测试程序" OFF)

# ============================================================
# MSVC 编译器设置
# ============================================================
if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra)
endif()

# ============================================================
# 源文件列表
# ============================================================
set(YOLO_SOURCES
    src/yolo_factory.cc
    src/backend/cpu_backend.cc
)

# ============================================================
# OpenCV 后端配置
# ============================================================
if(YOLO_WITH_OPENCV)
    find_package(OpenCV QUIET COMPONENTS core imgproc dnn)
    if(OpenCV_FOUND)
        list(APPEND YOLO_SOURCES src/backend/opencv_backend.cc)
        add_compile_definitions(YOLO_OPENCV_BACKEND=1)
        message(STATUS "OpenCV 后端: 已启用 (${OpenCV_VERSION})")
    else()
        message(WARNING "未找到 OpenCV，禁用 OpenCV 后端")
    endif()
endif()

# ============================================================
# OpenVINO 后端配置
# ============================================================
if(YOLO_WITH_OPENVINO)
    find_package(OpenVINO QUIET)
    if(OpenVINO_FOUND)
        list(APPEND YOLO_SOURCES src/backend/openvino_backend.cc)
        add_compile_definitions(YOLO_OPENVINO_BACKEND=1)
        message(STATUS "OpenVINO 后端: 已启用")
    else()
        message(WARNING "未找到 OpenVINO，禁用 OpenVINO 后端")
    endif()
endif()

# ============================================================
# 创建共享库
# ============================================================
add_library(yolo_interface SHARED ${YOLO_SOURCES})
target_compile_definitions(yolo_interface PRIVATE YOLO_EXPORTS)

target_include_directories(yolo_interface PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(OpenCV_FOUND)
    target_link_libraries(yolo_interface PUBLIC ${OpenCV_LIBS})
endif()

# ============================================================
# 设置输出目录
# ============================================================
set_target_properties(yolo_interface PROPERTIES
    OUTPUT_NAME "yolo_interface"
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# ============================================================
# 安装目标
# ============================================================
install(TARGETS yolo_interface
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# ============================================================
# 示例和测试
# ============================================================
if(YOLO_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(YOLO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()