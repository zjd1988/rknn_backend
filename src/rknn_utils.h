// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "rknn_api.h"
#include "triton/core/tritonserver.h"

#define RESPOND_ALL_AND_RETURN_IF_ERROR(RESPONSES, RESPONSES_COUNT, X) \
  do {                                                                 \
    TRITONSERVER_Error* raarie_err__ = (X);                            \
    if (raarie_err__ != nullptr) {                                     \
      SendErrorForResponses(RESPONSES, RESPONSES_COUNT, raarie_err__); \
      return;                                                          \
    }                                                                  \
  } while (false)

#define RETURN_IF_TRITONRKNN_ERROR(ERR)                                      \
  do {                                                                       \
    TRITONRKNN_Error* error__ = (ERR);                                       \
    if (error__ != nullptr) {                                                \
      auto status =                                                          \
          TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, error__->msg_); \
      TRITONRKNN_ErrorDelete(error__);                                       \
      return status;                                                         \
    }                                                                        \
  } while (false)

#define THROW_IF_TRITONRKNN_ERROR(X)           \
  do {                                         \
    TRITONRKNN_Error* tie_err__ = (X);         \
    if (tie_err__ != nullptr) {                \
      throw TRITONRKNN_Exception(tie_err__);   \
    }                                          \
  } while (false)

typedef struct 
{
    char* msg_;
} TRITONRKNN_Error;

struct TRITONRKNN_Exception
{
    TRITONRKNN_Exception(TRITONRKNN_Error* err) : err_(err) {}
    TRITONRKNN_Error* err_;
};

TRITONRKNN_Error* TRITONRKNN_ErrorNew(const std::string& str);

void TRITONRKNN_ErrorDelete(TRITONRKNN_Error* error);

// TRITONRKNN TYPE
// TODO: Full all possible type?
typedef enum
{
    TRITONRKNN_TYPE_FP32,
    TRITONRKNN_TYPE_FP16,
    TRITONRKNN_TYPE_INT8,
    TRITONRKNN_TYPE_UINT8,
    TRITONRKNN_TYPE_INT16,
    TRITONRKNN_TYPE_INVALID
} TRITONRKNN_DataType;

typedef enum
{
    TRITONRKNN_FORMAT_NCHW,
    TRITONRKNN_FORMAT_NHWC,
    TRITONRKNN_FORMAT_INVALID
} TRITONRKNN_FormatType;

// TODO: Full all possible core mask?
typedef enum
{
    TRITONRKNN_CORE_AUTO,
    TRITONRKNN_CORE_0,
    TRITONRKNN_CORE_1,
    TRITONRKNN_CORE_2,
    TRITONRKNN_CORE_INVALID
} TRITONRKNN_CoreMaskType;

// TRITONRKNN SHAPE
class TRITONRKNN_Shape 
{
public:
    using value_type = int64_t;

    TRITONRKNN_Shape() = default;
    TRITONRKNN_Shape(const std::string& str);
    template <typename T>
    TRITONRKNN_Shape(const std::vector<T>& shape);
    size_t NumElements() const { return numel_; };

    std::vector<int32_t> CompatibleShape() const;
    std::vector<value_type> Shape() const { return shape_; };

private:
    std::vector<value_type> shape_;
    size_t numel_;
};

TRITONRKNN_DataType ConvertDataType(TRITONSERVER_DataType dtype);

TRITONRKNN_DataType ConvertDataType(rknn_tensor_type dtype);

TRITONRKNN_DataType ConvertDataType(const std::string& dtype);

TRITONSERVER_DataType ConvertDataType(TRITONRKNN_DataType dtype);

std::string RknnDataTypeToModelConfigDataType(rknn_tensor_type data_type);

rknn_core_mask ConvertCoreMask(TRITONRKNN_CoreMaskType core_mask);

size_t TRITONRKNN_DataTypeByteSize(TRITONRKNN_DataType dtype);

// TRITON RKNN CONFIG
class TRITONRKNN_Config
{
public:
    TRITONRKNN_Config();
    uint8_t  want_float;
    uint8_t  pass_through;
    uint32_t flag;
};

// }}}
