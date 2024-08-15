##################################
## link//
##  onnxruntime: https://github.com/microsoft/onnxruntime/releases


##################################



cmake_minimum_required(VERSION 3.10)


project(ort_pct_demo)

## OPENCV ###################################################################
# set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../opencv480/x64/vc16/lib")
set(OpenCV_DIR "D:/source/code/work/3rdparty/opencv480/x64/vc16/lib")
# find_package(OpenCV REQUIRED core imgproc)
find_package(OpenCV REQUIRED)

# 打印相关信息
if (OpenCV_FOUND)
message("CMAKE_SOURCE_DIR is ====> ${CMAKE_SOURCE_DIR}")
message("OpenCV_DIR is --> ${OpenCV_DIR}")
message(STATUS "include path ${OpenCV_INCLUDE_DIRS}")
message(STATUS "opencv library status: ")
message(STATUS "version ${OpenCV_VERSION}")
message(STATUS "libraries: ${OpenCV_LIBS}")
else ()
message("OpenCV_DIR is --> ${OpenCV_DIR}")
message(FATAL_ERROR "Could not find OpenCV")
endif ()

include_directories(${OpenCV_INCLUDE_DIRS})
## OPENCV ###################################################################


## 增加头文件 ###############################################################
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ncnn20230816/include/ncnn)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../onnxruntime_win1.15.0/include
    
)
## 增加头文件  end


## 增加库文件 ###############################################################
link_directories(ort_demo
${CMAKE_CURRENT_SOURCE_DIR}/../../../onnxruntime_win1.15.0/lib

)
## 增加库文件   end



## 打包的库文件 ###############################################################
# add_library(ort_demo
#     ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp

# )
## 打包的库文件   end


## 库文件 ###############################################################
link_libraries( 
    ${OpenCV_LIBS}
    onnxruntime 
    # onnxruntime_providers_shared
    
)
## 库文件   end


##  编译文件 ###############################################################
add_executable(ort_demo
${CMAKE_CURRENT_SOURCE_DIR}/src/yolov5.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp

)
##  编译文件   end




# 设置可执行文件的输出目录
set_target_properties(ort_demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
# set_target_properties(ort_demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")

# 设置可执行文件的安装目录
install(TARGETS ort_demo DESTINATION bin)




