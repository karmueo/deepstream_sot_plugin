cmake_minimum_required(VERSION 3.16)
project(sot)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(DS_ROOT_DIR /opt/nvidia/deepstream/deepstream)
set(DS_LIBS_DIR ${DS_ROOT_DIR}/lib)
set(DS_INCLUDES_DIR ${DS_ROOT_DIR}/sources/includes)

find_package(GLib)
find_package(OpenCV)

include_directories(${GLIB2_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${DS_INCLUDES_DIR})
include_directories(${CMAKE_SOURCE_DIR}/includes)
include_directories(/usr/local/cuda/targets/x86_64-linux/include/)

file(GLOB TRACK CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -fPIC -g")
set(CMAKE_CUDA_FLAGS "--compiler-options -fPIC -g")

link_directories(${DS_LIBS_DIR})
add_library(${PROJECT_NAME} SHARED ${TRACK})

target_link_libraries(${PROJECT_NAME} nvds_meta ${OpenCV_LIBS})

# 添加安装
set(GST_PLUGINS_DIR "/opt/nvidia/deepstream/deepstream/lib")
install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION ${GST_PLUGINS_DIR}
        COMPONENT runtime
)

install(CODE [[
        set(SRC_CONFIG "${CMAKE_SOURCE_DIR}/config_sot.yml")
        if(NOT EXISTS "${SRC_CONFIG}")
                message(FATAL_ERROR "Missing config file: ${SRC_CONFIG}")
        endif()
        set(SRC_MODELS "${CMAKE_SOURCE_DIR}/models")
        if(NOT IS_DIRECTORY "${SRC_MODELS}")
                message(FATAL_ERROR "Missing models directory: ${SRC_MODELS}")
        endif()
        set(DEST_DIR "/opt/nvidia/deepstream/deepstream/sot_plugin")
        set(DEST_CONFIG "${DEST_DIR}/config/config_sot.yml")
        set(DEST_MODELS_DIR "${DEST_DIR}/models")
        file(MAKE_DIRECTORY "${DEST_DIR}/config")
        if(EXISTS "${DEST_CONFIG}")
                if(IS_SYMLINK "${DEST_CONFIG}")
                        file(REMOVE "${DEST_CONFIG}")
                else()
                        message(FATAL_ERROR "Existing file at ${DEST_CONFIG} is not a symlink; please remove it manually.")
                endif()
        endif()
        file(CREATE_LINK "${SRC_CONFIG}" "${DEST_CONFIG}" SYMBOLIC)
        file(CREATE_LINK "${SRC_MODELS}" "${DEST_MODELS_DIR}" SYMBOLIC)
]])