cmake_minimum_required(VERSION 3.16)
project(sot)

get_filename_component(REPOSITORY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
add_compile_options("-ffile-prefix-map=${REPOSITORY_ROOT}=.")

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(DS_ROOT_DIR /opt/nvidia/deepstream/deepstream)
set(DS_LIBS_DIR ${DS_ROOT_DIR}/lib)
set(DS_INCLUDES_DIR ${DS_ROOT_DIR}/sources/includes)

find_package(GLib)
find_package(yaml-cpp REQUIRED)

# SOT 统一使用 /usr/local 下源码编译的 OpenCV 4.10。
# EXACT rejects the incomplete JetPack OpenCV 4.8 package under /usr.
find_package(OpenCV 4.10 EXACT REQUIRED COMPONENTS core imgproc)

include_directories(${GLIB2_INCLUDE_DIR})
include_directories(${DS_INCLUDES_DIR})
include_directories(${CMAKE_SOURCE_DIR}/includes)
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    include_directories(/usr/local/cuda/targets/x86_64-linux/include/)
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include/)
endif()

file(GLOB TRACK CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -fPIC -g")
set(CMAKE_CUDA_FLAGS "--compiler-options -fPIC -g")

link_directories(${DS_LIBS_DIR})
add_library(${PROJECT_NAME} SHARED ${TRACK})

find_library(CUDART_LIB cudart
    PATHS /usr/local/cuda/lib64
          /usr/local/cuda/targets/x86_64-linux/lib
          /usr/local/cuda/targets/aarch64-linux/lib
)
if(NOT CUDART_LIB)
    message(FATAL_ERROR "Could not find cudart library. Check CUDA installation.")
endif()

find_library(NVTOOLSEXT_LIB nvToolsExt
    PATHS /usr/local/cuda/lib64
          /usr/local/cuda/targets/x86_64-linux/lib
          /usr/local/cuda/targets/aarch64-linux/lib
)
if(NOT NVTOOLSEXT_LIB)
    message(FATAL_ERROR "Could not find nvToolsExt library. Check CUDA installation.")
endif()

target_link_libraries(${PROJECT_NAME} nvds_meta opencv_core opencv_imgproc yaml-cpp nvinfer
                      ${CUDART_LIB} ${NVTOOLSEXT_LIB})
# 部署时 OpenCV 运行库与 libsot.so 同放在 lib/ 下。RUNPATH 不会从主程序
# 传递到 libsot 的依赖，因此 libsot 自身只保留 $ORIGIN；开发时依赖
# ldconfig 中的 /usr/local/lib。
set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "$ORIGIN"
)

option(SOT_BUILD_TESTS "Build tests" OFF)
if(SOT_BUILD_TESTS)
    enable_testing()
    find_package(Catch2 REQUIRED)
    # 测试程序需要读图、读写视频和窗口函数，libsot 本身只依赖 core/imgproc。
    find_package(OpenCV 4.10 EXACT REQUIRED
        COMPONENTS core imgproc imgcodecs videoio highgui)
    set(SOT_TEST_OPENCV_LIBRARIES
        opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio
        opencv_highgui)
    add_executable(test_track tests/test_track.cpp)
    target_link_libraries(test_track PRIVATE ${PROJECT_NAME} nvds_meta ${SOT_TEST_OPENCV_LIBRARIES} Catch2::Catch2WithMain)
    set_target_properties(test_track PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/lib"
    )
    target_compile_definitions(test_track PRIVATE
        TRACK_TEST_CONFIG_PATH="${CMAKE_SOURCE_DIR}/tests/track_test_config.yml"
        TRACK_TEST_REPO_ROOT="${CMAKE_SOURCE_DIR}"
    )
    
    # 使用 Catch2 的自动测试发现，让 CTest 发现所有 Catch2 测试用例
    include(Catch)
    catch_discover_tests(test_track)

    add_executable(test_track_video tests/test_track_video.cpp)
    target_link_libraries(test_track_video PRIVATE ${PROJECT_NAME} nvds_meta ${SOT_TEST_OPENCV_LIBRARIES})
    set_target_properties(test_track_video PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/lib"
    )
    target_compile_definitions(test_track_video PRIVATE
        TRACK_VIDEO_TEST_CONFIG_PATH="${CMAKE_SOURCE_DIR}/tests/track_video_test_config.yml"
    )
endif()

# 添加安装
set(GST_PLUGINS_DIR "/opt/nvidia/deepstream/deepstream/lib")
install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION ${GST_PLUGINS_DIR}
        COMPONENT runtime
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/install_symlinks.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/install_symlinks.cmake
        @ONLY
)

install(SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/install_symlinks.cmake)
