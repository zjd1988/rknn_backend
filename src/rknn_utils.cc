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

#include "rknn_utils.h"
#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <sstream>

template TRITONRKNN_Shape::TRITONRKNN_Shape(const std::vector<int64_t>& shape);
template TRITONRKNN_Shape::TRITONRKNN_Shape(const std::vector<int32_t>& shape);

template <typename T>
TRITONRKNN_Shape::TRITONRKNN_Shape(const std::vector<T>& shape)
{
    shape_ = std::vector<value_type>(shape.cbegin(), shape.cend());
    numel_ = std::accumulate(shape_.cbegin(), shape_.cend(), 1, std::multiplies<value_type>());
}

TRITONRKNN_Shape::TRITONRKNN_Shape(const std::string& str)
{
    std::vector<std::string> str_shape;
    std::istringstream in(str);
    std::copy(std::istream_iterator<std::string>(in), std::istream_iterator<std::string>(), std::back_inserter(str_shape));

    std::transform(str_shape.cbegin(), str_shape.cend(), std::back_inserter(shape_),
        [](const std::string& str) -> value_type {
            return static_cast<value_type>(std::stoll(str));
        });
}

std::vector<int32_t> TRITONRKNN_Shape::CompatibleShape() const
{
    return std::vector<int32_t>(shape_.cbegin(), shape_.cend());
}

TRITONRKNN_DataType ConvertDataType(TRITONSERVER_DataType dtype)
{
    switch (dtype)
    {
        case TRITONSERVER_TYPE_INVALID:
            return TRITONRKNN_TYPE_INVALID;
        case TRITONSERVER_TYPE_UINT8:
            return TRITONRKNN_TYPE_UINT8;
        case TRITONSERVER_TYPE_INT8:
            return TRITONRKNN_TYPE_INT8;
        case TRITONSERVER_TYPE_INT16:
            return TRITONRKNN_TYPE_INT16;
        case TRITONSERVER_TYPE_FP32:
            return TRITONRKNN_TYPE_FP32;
        case TRITONSERVER_TYPE_FP16:
            return TRITONRKNN_TYPE_FP16;
        default:
            break;
    }
    return TRITONRKNN_TYPE_INVALID;
}

TRITONSERVER_DataType ConvertDataType(TRITONRKNN_DataType dtype)
{
    switch (dtype)
    {
        case TRITONRKNN_TYPE_INVALID:
            return TRITONSERVER_TYPE_INVALID;
        case TRITONRKNN_TYPE_UINT8:
            return TRITONSERVER_TYPE_UINT8;
        case TRITONRKNN_TYPE_INT8:
            return TRITONSERVER_TYPE_INT8;
        case TRITONRKNN_TYPE_INT16:
            return TRITONSERVER_TYPE_INT16;
        case TRITONRKNN_TYPE_FP32:
            return TRITONSERVER_TYPE_FP32;
        case TRITONRKNN_TYPE_FP16:
            return TRITONSERVER_TYPE_FP16;
        default:
            break;
    }
    return TRITONSERVER_TYPE_INVALID;
}

TRITONRKNN_DataType ConvertDataType(const std::string& dtype)
{
  if (dtype == "TYPE_INVALID") {
    return TRITONRKNN_DataType::TRITONRKNN_TYPE_INVALID;
  } else if (dtype == "TYPE_FP32") {
    return TRITONRKNN_DataType::TRITONRKNN_TYPE_FP32;
  } else if (dtype == "TYPE_UINT8") {
    return TRITONRKNN_DataType::TRITONRKNN_TYPE_UINT8;
  } else if (dtype == "TYPE_INT8") {
    return TRITONRKNN_DataType::TRITONRKNN_TYPE_INT8;
  } else if (dtype == "TYPE_INT16") {
    return TRITONRKNN_DataType::TRITONRKNN_TYPE_INT16;
  } else if (dtype == "TYPE_FP16") {
    return TRITONRKNN_DataType::TRITONRKNN_TYPE_FP16;
  } 
  return TRITONRKNN_DataType::TRITONRKNN_TYPE_INVALID;
}

TRITONRKNN_DataType ConvertDataType(rknn_tensor_type dtype)
{
    switch (dtype)
    {
        case RKNN_TENSOR_FLOAT32:
            return TRITONRKNN_TYPE_FP32;
        case RKNN_TENSOR_FLOAT16:
            return TRITONRKNN_TYPE_FP16;
        case RKNN_TENSOR_INT8:
            return TRITONRKNN_TYPE_INT8;
        case RKNN_TENSOR_UINT8:
            return TRITONRKNN_TYPE_UINT8;
        case RKNN_TENSOR_INT16:
            return TRITONRKNN_TYPE_INT16;
        default:
            break;
  }
  return TRITONRKNN_TYPE_INVALID;
}

std::string RknnDataTypeToModelConfigDataType(rknn_tensor_type data_type)
{
    if (data_type == RKNN_TENSOR_FLOAT32)
    {
        return "TYPE_FP32";
    } 
    else if (data_type == RKNN_TENSOR_FLOAT16)
    {
        return "TYPE_FP16";
    }
    else if (data_type == RKNN_TENSOR_INT8)
    {
        return "TYPE_INT8";
    }
    else if (data_type == RKNN_TENSOR_UINT8)
    {
        return "TYPE_UINT8";
    }
    else if (data_type == RKNN_TENSOR_INT16)
    {
        return "TYPE_INT16";
    }
    return "TYPE_INVALID";
}

rknn_core_mask ConvertCoreMask(TRITONRKNN_CoreMaskType core_mask)
{
    switch (core_mask)
    {
        case TRITONRKNN_CORE_AUTO:
            return RKNN_NPU_CORE_AUTO;
        case TRITONRKNN_CORE_0:
            return RKNN_NPU_CORE_0;
        case TRITONRKNN_CORE_1:
            return RKNN_NPU_CORE_1;
        case TRITONRKNN_CORE_2:
            return RKNN_NPU_CORE_2;
        default:
            break;
    }
    return RKNN_NPU_CORE_UNDEFINED;
}

size_t TRITONRKNN_DataTypeByteSize(TRITONRKNN_DataType dtype)
{
    switch (dtype)
    {
        case TRITONRKNN_DataType::TRITONRKNN_TYPE_FP32:
            return sizeof(float);
        case TRITONRKNN_DataType::TRITONRKNN_TYPE_FP16:
            return sizeof(int16_t);
        case TRITONRKNN_DataType::TRITONRKNN_TYPE_UINT8:
            return sizeof(uint8_t);
        case TRITONRKNN_DataType::TRITONRKNN_TYPE_INT8:
            return sizeof(int8_t);
        case TRITONRKNN_DataType::TRITONRKNN_TYPE_INT16:
            return sizeof(int16_t);
        default:
            break;
    }
    return 0;  // Should not happened, TODO: Error handling
}

/* Error message */
TRITONRKNN_Error* TRITONRKNN_ErrorNew(const std::string& str)
{
    TRITONRKNN_Error* error = new TRITONRKNN_Error();
    error->msg_ = new char[str.size() + 1];
    std::strcpy(error->msg_, str.c_str());
    return error;
}

void TRITONRKNN_ErrorDelete(TRITONRKNN_Error* error)
{
    if (error == nullptr)
    {
        return;
    }

    delete[] error->msg_;
    delete error;
}

TRITONRKNN_Config::TRITONRKNN_Config() : want_float(1), pass_through(0), flag(0)
{
}

// }}}
