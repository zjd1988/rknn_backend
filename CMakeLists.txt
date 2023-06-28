# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.17)

project(trironrknnbackend LANGUAGES C CXX)

set(RKNN_INFERENCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rknpu2" CACHE PATH "Paths to rknn Inference Directory. Multiple paths may be specified by sparating them with a semicolon.")
set(RKNN_INCLUDE_PATHS "${RKNN_INFERENCE_DIR}/include"
  CACHE PATH "Paths to rknn Inference includes. Multiple paths may be specified by sparating them with a semicolon.")
set(RKNN_LIB_PATHS "${RKNN_INFERENCE_DIR}/lib"
  CACHE PATH "Paths to rknn Inference libraries. Multiple paths may be specified by sparating them with a semicolon.")
set(RKNN_LIB_NAME "rknn_api")

set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")

include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

configure_file(src/libtriton_rknn.ldscript libtriton_rknn.ldscript COPYONLY)

add_library(
  triton-rknn-backend SHARED
  src/rknn.cc
  src/rknn_utils.cc
)

add_library(
  TritonRknnBackend::triton-rknn-backend ALIAS triton-rknn-backend
)

target_include_directories(triton-rknn-backend PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(triton-rknn-backend PRIVATE 
  ${RKNN_INCLUDE_PATHS}
)

target_link_libraries(triton-rknn-backend PRIVATE 
  "-L${RKNN_LIB_PATHS} -l${RKNN_LIB_NAME}"
)

target_compile_features(triton-rknn-backend PRIVATE cxx_std_11)
target_compile_options(triton-rknn-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
)

set_target_properties(triton-rknn-backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME triton_rknn
  SKIP_BUILD_RPATH TRUE
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_rknn.ldscript
  LINK_FLAGS "-Wl,--version-script libtriton_rknn.ldscript"
)

target_link_libraries(triton-rknn-backend PRIVATE
  triton-backend-utils    # from repo-backend
  triton-core-serverstub  # from repo-core
)

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonRknnBackend)

install(
  TARGETS
    triton-rknn-backend
  EXPORT
    triton-rknn-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/rknn
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/rknn
)

set(RKNN_LIBS 
  "librknn_api.so"
  "librknnrt.so")
FOREACH(rklib ${RKNN_LIBS})
  set(RK_LIB_PATHS ${RK_LIB_PATHS} "${RKNN_LIB_PATHS}/${rklib}")
ENDFOREACH(rklib)

install(FILES ${RK_LIB_PATHS}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/rknn
)

FOREACH(rklib ${PT_LIBS})
  install(
    CODE
      "EXECUTE_PROCESS(
        COMMAND patchelf --set-rpath \$ORIGIN ${rklib}
        RESULT_VARIABLE PATCHELF_STATUS
        WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/backends/rknn)
      if(PATCHELF_STATUS AND NOT PATCHELF_STATUS EQUAL 0)
        message(FATAL_ERROR \"FAILED: to run patchelf\")
      endif()"
  )
ENDFOREACH(rklib)


install(
  EXPORT
    triton-rknn-backend-targets
  FILE
    TritonRknnBackendTargets.cmake
  NAMESPACE
    TritonRknnBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonRknnBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonRknnBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonRknnBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-rknn-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonRknnBackendTargets.cmake
  NAMESPACE TritonRknnBackend::
)

export(PACKAGE TritonRknnBackend)
