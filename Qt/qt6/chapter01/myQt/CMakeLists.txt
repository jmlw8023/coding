cmake_minimum_required(VERSION 3.5)

project(myQt VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

find_package(Qt6 REQUIRED COMPONENTS SerialPort)
find_package(Qt6 REQUIRED COMPONENTS PrintSupport)
find_package(Qt6Charts)

find_package(Qt6 COMPONENTS Multimedia REQUIRED)
find_package(Qt6 COMPONENTS MultimediaWidgets REQUIRED)

## msvc编译
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/opencv480/x64/vc16/lib")
# set(OpenCV_DIR "D:/source/code/work/3rdparty/opencv480/x64/vc16/lib")

# mingw编译
# set(OpenCV_DIR "D:/source/code/work/3rdparty/opencv480_mingw64")
# set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/opencv480_mingw64")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

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




set(PROJECT_SOURCES
        main.cpp
        mainwindow.cpp
        mainwindow.h
        mainwindow.ui
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(myQt
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
        imgprocess.h imgprocess.cpp
    )
# Define target properties for Android with Qt 6 as:
#    set_property(TARGET myQt APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
#                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
# For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation
else()
    if(ANDROID)
        add_library(myQt SHARED
            ${PROJECT_SOURCES}
        )
# Define properties for Android with Qt 5 after find_package() calls as:
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(myQt
            ${PROJECT_SOURCES}
        )
    endif()
endif()

target_link_libraries(myQt PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    Qt${QT_VERSION_MAJOR}::PrintSupport
    Qt${QT_VERSION_MAJOR}::Charts
    Qt${QT_VERSION_MAJOR}::SerialPort

    Qt${QT_VERSION_MAJOR}::Multimedia
    Qt${QT_VERSION_MAJOR}::MultimediaWidgets

    ${OpenCV_LIBS}

)




# Qt for iOS sets MACOSX_BUNDLE_GUI_IDENTIFIER automatically since Qt 6.1.
# If you are developing for iOS or macOS you should consider setting an
# explicit, fixed bundle identifier manually though.
if(${QT_VERSION} VERSION_LESS 6.1.0)
  set(BUNDLE_ID_OPTION MACOSX_BUNDLE_GUI_IDENTIFIER com.example.myQt)
endif()
set_target_properties(myQt PROPERTIES
    ${BUNDLE_ID_OPTION}
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

include(GNUInstallDirs)
install(TARGETS myQt
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(myQt)
endif()
