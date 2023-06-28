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

#include <map>
#include <algorithm>
#include <memory>
#include <numeric>
#include <fstream>

#include "rknn_utils.h"
#include "rknn_api.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"

std::map<int, TRITONRKNN_CoreMaskType> gCoreMaskMap = {
    {-1, TRITONRKNN_CORE_AUTO},
    {0, TRITONRKNN_CORE_0},
    {1, TRITONRKNN_CORE_1},
    {2, TRITONRKNN_CORE_2},
};

// rknn tensor Wrapper
struct TRITONRKNN_Tensor;

class TensorImpl
{
public:
    TensorImpl(const char* name, TRITONRKNN_DataType dtype, const TRITONRKNN_Shape& shape);
    ~TensorImpl() = default;

    const std::string& Name() const { return m_name; }
    TRITONRKNN_DataType DataType() const { return m_dtype; }
    TRITONRKNN_Shape Shape() const { return m_shape; }

    template<class T>
    T* Base() const { return (T*)(m_base.get()); }

    size_t ByteSize() const { return m_byte_size; }

    rknn_tensor_type RknnTensorType();
    rknn_tensor_format RknnFormatType();

private:
    const std::string                m_name;
    const TRITONRKNN_DataType        m_dtype;
    const TRITONRKNN_Shape           m_shape;
    size_t                           m_byte_size;
    std::shared_ptr<char>            m_base;
};

TensorImpl::TensorImpl(const char* name, TRITONRKNN_DataType dtype, const TRITONRKNN_Shape& shape)
    : m_name(name), m_dtype(dtype), m_shape(shape)
{
    m_byte_size = shape.NumElements() * TRITONRKNN_DataTypeByteSize(dtype);
    m_base.reset(new char[m_byte_size], std::default_delete<char[]>());
    if (nullptr == m_base.get())
    {
        // TRITONRKNN_Error* error = TRITONRKNN_ErrorNew(std::string("rknpu engine malloc mem for tensor ") + m_name + " fail!");
        // THROW_IF_TRITONRKNN_ERROR(error);
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("malloc mem for tensor ") + m_name + " fail!").c_str());
    }
}

rknn_tensor_type TensorImpl::RknnTensorType()
{
    switch (m_dtype)
    {
        case TRITONRKNN_TYPE_INVALID:
            return RKNN_TENSOR_TYPE_MAX;
        case TRITONRKNN_TYPE_UINT8:
            return RKNN_TENSOR_UINT8;
        case TRITONRKNN_TYPE_INT8:
            return RKNN_TENSOR_INT8;
        case TRITONRKNN_TYPE_INT16:
            return RKNN_TENSOR_INT16;
        case TRITONRKNN_TYPE_FP32:
            return RKNN_TENSOR_FLOAT32;
        case TRITONRKNN_TYPE_FP16:
            return RKNN_TENSOR_FLOAT16;
        default:
            break;
    }
    return RKNN_TENSOR_TYPE_MAX;
}

rknn_tensor_format TensorImpl::RknnFormatType()
{
    rknn_tensor_format fmt_type = RKNN_TENSOR_UNDEFINED;
    auto tensor_shape = m_shape.Shape();
    if (4 == tensor_shape.size() && (1 == tensor_shape[1] || 3 == tensor_shape[1]))
        fmt_type = RKNN_TENSOR_NCHW;
    else if (4 == tensor_shape.size() && (1 == tensor_shape[3] || 3 == tensor_shape[3]))
        fmt_type = RKNN_TENSOR_NHWC;
    
    return fmt_type;
}

TRITONSERVER_Error* TRITONRKNN_TensorCreate(TRITONRKNN_Tensor** tensor, const char* name, 
    TRITONSERVER_DataType triton_dtype, const std::vector<int64_t> shape)
{
    try
    {
        TRITONRKNN_DataType data_type = ConvertDataType(triton_dtype);
        TRITONRKNN_Shape    rknn_shape(shape);
        TensorImpl* tensor_impl = new TensorImpl(name, data_type, rknn_shape);
        *tensor = reinterpret_cast<TRITONRKNN_Tensor*>(tensor_impl);
    }
    catch (const TRITONRKNN_Exception& ex)
    {
        RETURN_IF_TRITONRKNN_ERROR(ex.err_);
    }
    return nullptr;  
}

void TRITONRKNN_TensorDelete(TRITONRKNN_Tensor* tensor)
{
    if (tensor != nullptr)
    {
        TensorImpl* ti = reinterpret_cast<TensorImpl*>(tensor);
        delete ti;
    }
}

char* TRITONRKNN_TensorData(TRITONRKNN_Tensor* tensor)
{
    TensorImpl* t = reinterpret_cast<TensorImpl*>(tensor);
    return t->Base<char>();
}

size_t TRITONRKNN_TensorDataByteSize(TRITONRKNN_Tensor* tensor)
{
    TensorImpl* t = reinterpret_cast<TensorImpl*>(tensor);
    return t->ByteSize();
}

TRITONRKNN_DataType TRITONRKNN_TensorDataType(TRITONRKNN_Tensor* tensor)
{
    TensorImpl* t = reinterpret_cast<TensorImpl*>(tensor);
    return t->DataType();
}

TRITONRKNN_Shape TRITONRKNN_TensorShape(TRITONRKNN_Tensor* tensor)
{
    TensorImpl* t = reinterpret_cast<TensorImpl*>(tensor);
    return t->Shape();
}

// rknn model Wrapper
struct TRITONRKNN_Model;

class ModelImpl
{
public:
    ModelImpl(const char* model_name, const char* model_path, TRITONRKNN_Config* config, int device_id);
    ~ModelImpl();
    TRITONRKNN_Error* Run();
    TRITONRKNN_Error* SetInputTensors(std::map<std::string, std::shared_ptr<TRITONRKNN_Tensor>>& input_tensors);
    TRITONRKNN_Error* GetOutputTensors(std::map<std::string, std::shared_ptr<TRITONRKNN_Tensor>>& output_tensors);
    // TRITONRKNN_Error* ZeroCopyRun();

    // engine funcs
    int GetEngineTensorInfos(std::vector<rknn_tensor_attr> &input_tensor_infos, std::vector<rknn_tensor_attr> &output_tensor_infos);
    int GetEngineInputNames(std::vector<std::string> &input_tensor_names);
    int GetEngineOutputNames(std::vector<std::string> &output_tensor_names);

private:
    void PrintTensorInfo(rknn_tensor_attr &attr);
    void PrintEngineInfo();

private:
    // TODO(wilber): unique_ptr?
    std::string                                                m_model_name;
    rknn_context                                               m_engine_ctx = 0;
    std::vector<std::string>                                   m_input_tensor_names;
    std::vector<std::string>                                   m_output_tensor_names;
    bool                                                       m_pass_through = false;
    bool                                                       m_want_float = true;
};

int ModelImpl::GetEngineTensorInfos(std::vector<rknn_tensor_attr> &input_tensor_infos, std::vector<rknn_tensor_attr> &output_tensor_infos)
{
    int ret = 0;
    if (0 == m_engine_ctx)
    {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("rknpu engine context is null").c_str()));
        return -1;
    }
    rknn_context context = m_engine_ctx;
    input_tensor_infos.clear();
    output_tensor_infos.clear();
    
    // get input/output tensor num
    rknn_input_output_num io_num;
    ret = rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("rknn_query RKNN_QUERY_IN_OUT_NUM fail! ret = " + std::to_string(ret)).c_str()));
        return -1;
    }

    // get input tensor attr
    for (uint32_t i = 0; i < io_num.n_input; i++)
    {
        rknn_tensor_attr input_attrs;
        memset(&input_attrs, 0, sizeof(input_attrs));
        input_attrs.index = i;
        ret = rknn_query(context, RKNN_QUERY_INPUT_ATTR, &input_attrs, sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("rknn_query RKNN_QUERY_INPUT_ATTR fail! ret = " + std::to_string(ret)).c_str()));
            return -1;
        }
        input_tensor_infos.push_back(input_attrs);
    }

    // get output tensor attr
    for (uint32_t i = 0; i < io_num.n_output; i++)
    {
        rknn_tensor_attr output_attrs;
        memset(&output_attrs, 0, sizeof(output_attrs));        
        output_attrs.index = i;
        ret = rknn_query(context, RKNN_QUERY_OUTPUT_ATTR, &output_attrs, sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("rknn_query RKNN_QUERY_OUTPUT_ATTR fail! ret = " + std::to_string(ret)).c_str()));
            return -1;
        }
        output_tensor_infos.push_back(output_attrs);
    }
    return ret;
}

void ModelImpl::PrintTensorInfo(rknn_tensor_attr &attr)
{
    std::string shape_str;
    for (uint32_t i = 0 ; i < attr.n_dims; i++)
    {
        shape_str += std::to_string(attr.dims[i]) + " ";
    }
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, ((std::string("index=") + std::to_string(attr.index) + 
        std::string(", name=") + attr.name + std::string(", n_dims=") + std::to_string(attr.n_dims) +
        std::string(", dims=") + shape_str + std::string(", n_elems=") + std::to_string(attr.n_elems) + 
        std::string(", size=") + std::to_string(attr.size) + std::string(", fmt=") + get_format_string(attr.fmt) +
        std::string(", type=") + get_type_string(attr.type) + std::string(", qnt_type=") + get_qnt_type_string(attr.qnt_type) + 
        std::string(", fl=") + std::to_string(attr.fl) + std::string(", zp=") + std::to_string(attr.zp) +
        std::string(", scale=") + std::to_string(attr.scale)).c_str()));
    return;
}

void ModelImpl::PrintEngineInfo()
{
    if (0 == m_engine_ctx)
    {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (m_model_name + std::string(" rknpu engine context is null")).c_str());
        return;
    }
    std::vector<rknn_tensor_attr> input_tensor_attrs;
    std::vector<rknn_tensor_attr> output_tensor_attrs;
    int ret = GetEngineTensorInfos(input_tensor_attrs, output_tensor_attrs);
    if (0 == ret)
    {
        // log Model Input tensor Info
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (m_model_name + std::string(" input tensors:")).c_str());
        for (size_t i = 0; i < input_tensor_attrs.size(); i++)
        {
            PrintTensorInfo(input_tensor_attrs[i]);
        }

        // log Model Output tensor Info
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (m_model_name + std::string(" output tensors:")).c_str());
        for (size_t i = 0; i < output_tensor_attrs.size(); i++)
        {
            PrintTensorInfo(output_tensor_attrs[i]);
        }
    }
    else
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (m_model_name + std::string(" get rknpu model tensor attr fail")).c_str());
    
    // Get custom string
    rknn_custom_string custom_string;
    ret = rknn_query(m_engine_ctx, RKNN_QUERY_CUSTOM_STRING, &custom_string, sizeof(custom_string));
    if (ret != RKNN_SUCC)
    {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, ((m_model_name + std::string(" get rknpu model custom string attr fail, ret=") + std::to_string(ret)).c_str()));
        return;
    }
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (m_model_name + std::string(" custom string: ") + std::string(custom_string.string)).c_str());
    return;
}

int ModelImpl::GetEngineInputNames(std::vector<std::string> &input_tensor_names)
{
    input_tensor_names.clear();
    if (0 == m_engine_ctx)
    {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("rknpu engine context is null").c_str()));
        return -1;
    }
    std::vector<rknn_tensor_attr> input_tensor_attrs;
    std::vector<rknn_tensor_attr> output_tensor_attrs;        
    int ret = GetEngineTensorInfos(input_tensor_attrs, output_tensor_attrs);
    if (0 == ret)
    {
        for (size_t i = 0; i < input_tensor_attrs.size(); i++)
        {
            rknn_tensor_attr tensor_attr = input_tensor_attrs[i];
            input_tensor_names.push_back(tensor_attr.name);
        }
    }
    else
    {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("get rknpu engine input tensor attr fail").c_str()));
        return -1;
    }
    return 0;
}

int ModelImpl::GetEngineOutputNames(std::vector<std::string> &output_tensor_names)
{
    output_tensor_names.clear();
    if (0 == m_engine_ctx)
    {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("rknpu engine context is null").c_str()));
        return -1;
    }
    std::vector<rknn_tensor_attr> input_tensor_attrs;
    std::vector<rknn_tensor_attr> output_tensor_attrs;
    int ret = GetEngineTensorInfos(input_tensor_attrs, output_tensor_attrs);
    if (0 == ret)
    {
        for (size_t i = 0; i < output_tensor_attrs.size(); i++)
        {
            rknn_tensor_attr tensor_attr = output_tensor_attrs[i];
            output_tensor_names.push_back(tensor_attr.name);
        }
    }
    else
    {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("get rknpu engine output tensor attr fail").c_str()));
    }
    return 0;
}

ModelImpl::ModelImpl(const char* model_name, const char* model_path, TRITONRKNN_Config* config, int device_id)
{
    m_want_float = (0 != config->want_float);
    m_pass_through = (0 != config->pass_through);
    m_model_name = model_name;
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (m_model_name + std::string(" config want_float: ") + 
        std::to_string(m_want_float) + std::string(", pass_through: ") + std::to_string(m_pass_through)).c_str());
    std::ifstream model_stream(model_path);
    if (!model_stream.is_open())
    {
        TRITONRKNN_Error* error = TRITONRKNN_ErrorNew(std::string("open rknn model file fail, ") + std::string(model_path));
        THROW_IF_TRITONRKNN_ERROR(error);
    }
    // get rknn model file len
    model_stream.seekg(0, std::ios::end);
    int file_len = model_stream.tellg();
    std::shared_ptr<char> model_data = std::shared_ptr<char>(new char[file_len], std::default_delete<char []>());

    // read rknn model file 
    model_stream.seekg(0, std::ios::beg);
    model_stream.read(model_data.get(), file_len);
    int read_len = !model_stream.bad() ? model_stream.gcount() : -1;
    model_stream.close();
    if (read_len != file_len)
    {
        TRITONRKNN_Error* error = TRITONRKNN_ErrorNew(std::string("read rknn model file fail, ") + std::string(model_path));
        THROW_IF_TRITONRKNN_ERROR(error);
    }
    int ret = rknn_init(&m_engine_ctx, model_data.get(), file_len, config->flag, NULL);
    if (0 != ret)
    {
        TRITONRKNN_Error* error = TRITONRKNN_ErrorNew(std::string("rknpu engine init fail!, ret = ") + std::to_string(ret));
        THROW_IF_TRITONRKNN_ERROR(error);
    }

    // for rk3588 need set core mask
    if (gCoreMaskMap.end() == gCoreMaskMap.find(device_id))
    {
        TRITONRKNN_Error* error = TRITONRKNN_ErrorNew(std::string("rknpu engine get invalid core mask = ") + std::to_string(device_id));
        THROW_IF_TRITONRKNN_ERROR(error);
    }
    rknn_core_mask core_mask = ConvertCoreMask(gCoreMaskMap[device_id]);
    ret = rknn_set_core_mask(m_engine_ctx, core_mask);
    if (0 != ret)
    {
        TRITONRKNN_Error* error = TRITONRKNN_ErrorNew(std::string("rknpu engine set core mask fail!, ret = ") + std::to_string(ret));
        THROW_IF_TRITONRKNN_ERROR(error);
    }

    // init engine tensors
    m_input_tensor_names.clear();
    m_output_tensor_names.clear();
    std::vector<rknn_tensor_attr> input_tensor_attrs;
    std::vector<rknn_tensor_attr> output_tensor_attrs;
    ret = GetEngineTensorInfos(input_tensor_attrs, output_tensor_attrs);
    if (0 != ret)
    {
        TRITONRKNN_Error* error = TRITONRKNN_ErrorNew(std::string("rknpu engine get tensors info fail!, ret = ") + std::to_string(ret));
        THROW_IF_TRITONRKNN_ERROR(error);
    }

    for (size_t i = 0; i < input_tensor_attrs.size(); i++)
    {
        std::string tensor_name = input_tensor_attrs[i].name;
        // TRITONRKNN_DataType tensor_dtype = ConvertDataType(input_tensor_attrs[i].type);
        // std::vector<int> shape(&input_tensor_attrs[i].dims[0], &input_tensor_attrs[i].dims[0] + input_tensor_attrs[i].n_dims);
        // TRITONRKNN_Shape tensor_shape(shape);
        // m_engine_tensors[tensor_name].reset(new TensorImpl(tensor_name.c_str(), tensor_dtype, tensor_shape));
        m_input_tensor_names.push_back(tensor_name);
    }

    for (size_t i = 0; i < output_tensor_attrs.size(); i++)
    {
        std::string tensor_name = output_tensor_attrs[i].name;
        // TRITONRKNN_DataType tensor_dtype = ConvertDataType(output_tensor_attrs[i].type);
        // if (config->want_float)
        //     tensor_dtype = TRITONRKNN_TYPE_FP32;
        // std::vector<int> shape(&output_tensor_attrs[i].dims[0], &output_tensor_attrs[i].dims[0] + output_tensor_attrs[i].n_dims);
        // TRITONRKNN_Shape tensor_shape(shape);
        // m_engine_tensors[tensor_name].reset(new TensorImpl(tensor_name.c_str(), tensor_dtype, tensor_shape));
        m_output_tensor_names.push_back(tensor_name);
    }
    PrintEngineInfo();

}

ModelImpl::~ModelImpl()
{
    if (0 != m_engine_ctx)
    {
        auto ret = rknn_destroy(m_engine_ctx);
        if (RKNN_SUCC == ret)
        {
            m_engine_ctx = 0;
        }
        else
        {
            // TRITONRKNN_Error* error = TRITONRKNN_ErrorNew(std::string("destroy rknpu engine context fail!, ret = ") + std::to_string(ret));
            // THROW_IF_TRITONRKNN_ERROR(error);
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("destroy rknpu engine context fail!, ret = ") + std::to_string(ret)).c_str());
        }
    }
}

TRITONRKNN_Error* ModelImpl::SetInputTensors(std::map<std::string, std::shared_ptr<TRITONRKNN_Tensor>>& input_tensors)
{
    if (0 == m_engine_ctx)
    {
        return TRITONRKNN_ErrorNew(std::string("rknpu engine context is null ptr"));
    }
    std::vector<rknn_tensor_attr> input_tensor_attrs;
    std::vector<rknn_tensor_attr> output_tensor_attrs;
    int ret = GetEngineTensorInfos(input_tensor_attrs, output_tensor_attrs);
    if (0 != ret)
    {
        return TRITONRKNN_ErrorNew(std::string("get rknpu engine tensors attr fail"));
    }
    std::vector<rknn_input> rk_input_tensors;
    for (size_t index = 0; index < input_tensor_attrs.size(); index++)
    {
        rknn_input current_input;
        std::string tensor_name = input_tensor_attrs[index].name;
        // check input tensor name
        if (input_tensors.find(tensor_name) == input_tensors.end())
        {
            return TRITONRKNN_ErrorNew(std::string("input tensor sented to rknpu engine should contain ") + tensor_name);
        }
        // check tensor type and format
        auto input_tensor = reinterpret_cast<TensorImpl*>(input_tensors[tensor_name].get());

        rknn_tensor_format expect_fmt = input_tensor_attrs[index].fmt;
        rknn_tensor_type expect_dtype = input_tensor_attrs[index].type;
        rknn_tensor_format input_format = input_tensor->RknnFormatType();
        rknn_tensor_type input_dtype = input_tensor->RknnTensorType();

        if (RKNN_TENSOR_UNDEFINED == input_format || RKNN_TENSOR_FORMAT_MAX == input_format)
        {
            return TRITONRKNN_ErrorNew(std::string("input tensor ") + tensor_name +
                std::string(" have invalid fromat ") + get_format_string(input_format));
        }

        if (RKNN_TENSOR_TYPE_MAX == input_dtype)
        {
            return TRITONRKNN_ErrorNew(std::string("input tensor ") + tensor_name +
                std::string(" have invalid datatype ") + get_type_string(input_dtype));
        }

        if (m_pass_through && expect_dtype != input_dtype)
        {
            return TRITONRKNN_ErrorNew(std::string("rknpu engine config pass-through true, input tensor ") + tensor_name +
                std::string(" have invalid datatype ") + get_type_string(input_dtype) + std::string(", expect ") + get_type_string(expect_dtype));
        }

        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("input tensor sented to rknpu engine ") + tensor_name +
            std::string("'s datatype is ") + get_type_string(input_dtype)).c_str());
        if (m_pass_through && expect_fmt != input_format)
        {
            return TRITONRKNN_ErrorNew(std::string("rknpu engine config pass-through true, input tensor ") + tensor_name +
                std::string(" have invalid fromat ") + get_format_string(input_format) + std::string(", expect ") + get_format_string(expect_fmt));
        }
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("input tensor sented to rknpu engine ") + tensor_name + 
            std::string("'s fromat is ") + get_format_string(input_format)).c_str());

        // check input tensor and rk tensor size
        int input_size = input_tensor->ByteSize();
        int expect_input_size = input_tensor_attrs[index].size;
        if (m_pass_through && input_size != expect_input_size)
        {
            return TRITONRKNN_ErrorNew(std::string("rknpu engine config pass-through true, input tensor size:") + 
                std::to_string(input_size) + std::string(" not equal to rk input tensor size:") + std::to_string(expect_input_size));
        }
        // check input tensor data valid
        if (input_tensor->Base<char>() == nullptr)
        {
            return TRITONRKNN_ErrorNew(std::string("rknpu engine input tensor ") + tensor_name + std::string("'s data is null ptr"));
        }

        // init rk input tensor
        memset(&current_input, 0, sizeof(current_input));
        current_input.index = index;
        if (m_pass_through)
            current_input.type = expect_dtype;
        else
            current_input.type = input_dtype;
        current_input.size = input_tensor->ByteSize();
        current_input.pass_through = m_pass_through;
        current_input.fmt = input_format;
        current_input.buf = input_tensor->Base<char>();
        rk_input_tensors.push_back(current_input);
    }
    ret = rknn_inputs_set(m_engine_ctx, input_tensor_attrs.size(), &rk_input_tensors[0]);
    if (RKNN_SUCC != ret)
    {
        return TRITONRKNN_ErrorNew(std::string("rknpu engine set input tensors fail! ret = ") + std::to_string(ret));
    }
    return nullptr;
}

TRITONRKNN_Error* ModelImpl::GetOutputTensors(std::map<std::string, std::shared_ptr<TRITONRKNN_Tensor>>& output_tensors)
{
    if (0 == m_engine_ctx)
    {
        return TRITONRKNN_ErrorNew(std::string("rknpu engine context is null ptr"));
    }
    std::vector<rknn_tensor_attr> input_tensor_attrs;
    std::vector<rknn_tensor_attr> output_tensor_attrs;
    int ret = GetEngineTensorInfos(input_tensor_attrs, output_tensor_attrs);
    if (0 != ret)
    {
        return TRITONRKNN_ErrorNew(std::string("get rknpu engine tensors attr fail"));
    }
    std::vector<rknn_output> rk_output_tensors;
    for (size_t index = 0; index < output_tensor_attrs.size(); index++)
    {
        rknn_output current_output;
        rknn_tensor_attr tensor_attr = output_tensor_attrs[index];
        std::string tensor_name = tensor_attr.name;
        // init output tensor
        TRITONRKNN_DataType tensor_dtype = ConvertDataType(tensor_attr.type);
        std::vector<int64_t> tensor_shape;
        for (uint32_t i = 0; i < tensor_attr.n_dims; i++)
        {
            tensor_shape.push_back(tensor_attr.dims[i]);
        }
        if (m_want_float)
            tensor_dtype = TRITONRKNN_TYPE_FP32;
        TensorImpl* output_tensor = new TensorImpl(tensor_name.c_str(), tensor_dtype, tensor_shape);
        TRITONRKNN_Tensor* tensor = reinterpret_cast<TRITONRKNN_Tensor*>(output_tensor);
        if (nullptr == tensor)
        {
            return TRITONRKNN_ErrorNew(std::string("rknpu engine create output tensor ") + tensor_name + std::string(" fail"));
        }        
        output_tensors[tensor_name].reset(tensor, TRITONRKNN_TensorDelete);

        // check output tensor valid
        if (output_tensor->Base<char>() == nullptr)
        {
            return TRITONRKNN_ErrorNew(std::string("rknpu engine output tensor ") + tensor_name + std::string(" data is nullptr"));
        }

        // init rk output tensor
        memset(&current_output, 0, sizeof(current_output));
        current_output.index = index;
        current_output.want_float = m_want_float;
        current_output.is_prealloc = 1;
        current_output.buf = output_tensor->Base<void>();
        current_output.size = output_tensor->ByteSize();
        rk_output_tensors.push_back(current_output);
    }

    // get rk output tensors
    ret = rknn_outputs_get(m_engine_ctx, rk_output_tensors.size(), &rk_output_tensors[0], NULL);
    if (RKNN_SUCC != ret)
    {
        return TRITONRKNN_ErrorNew(std::string("rknpu engine get output tensors fail! ret = ") + std::to_string(ret));
    }

    //no need to release output tensors, not malloc from npu memory
    ret = rknn_outputs_release(m_engine_ctx, rk_output_tensors.size(), &rk_output_tensors[0]);
    if (RKNN_SUCC != ret)
    {
        return TRITONRKNN_ErrorNew(std::string("release rknpu engine output tensor fail!, ret = ") + std::to_string(ret));
    }
    return nullptr;
}

TRITONRKNN_Error* ModelImpl::Run()
{
    if (0 == m_engine_ctx)
    {
        return TRITONRKNN_ErrorNew(std::string("rknpu engine context is null ptr"));
    }
    // run engine
    int ret = rknn_run(m_engine_ctx, nullptr);
    if (0 != ret)
    {
        return TRITONRKNN_ErrorNew(std::string("rknpu engine run fail. ret = ") + std::to_string(ret));
    }
    return nullptr;
}

TRITONSERVER_Error* TRITONRKNN_ModelCreate(TRITONRKNN_Model** model, const char* model_name, const char* model_path, TRITONRKNN_Config* config, int device_id)
{
    try
    {
        ModelImpl* model_impl = new ModelImpl(model_name, model_path, config, device_id);
        *model = reinterpret_cast<TRITONRKNN_Model*>(model_impl);
    }
    catch (const TRITONRKNN_Exception& ex)
    {
        RETURN_IF_TRITONRKNN_ERROR(ex.err_);
    }
    return nullptr;
}

void TRITONRKNN_ModelDelete(TRITONRKNN_Model* model)
{
    if (model != nullptr)
    {
        ModelImpl* mi = reinterpret_cast<ModelImpl*>(model);
        delete mi;
    }
}

TRITONSERVER_Error* TRITONRKNN_ModelSetInputTensors(TRITONRKNN_Model* model, std::map<std::string, std::shared_ptr<TRITONRKNN_Tensor>>& input_tensors)
{
    ModelImpl* m = reinterpret_cast<ModelImpl*>(model);
    RETURN_IF_TRITONRKNN_ERROR(m->SetInputTensors(input_tensors));
    return nullptr;
}

TRITONSERVER_Error* TRITONRKNN_ModelGetOutputTensors(TRITONRKNN_Model* model, std::map<std::string, std::shared_ptr<TRITONRKNN_Tensor>>& output_tensors)
{
    ModelImpl* m = reinterpret_cast<ModelImpl*>(model);
    RETURN_IF_TRITONRKNN_ERROR(m->GetOutputTensors(output_tensors));
    return nullptr;
}

TRITONSERVER_Error* TRITONRKNN_ModelRun(TRITONRKNN_Model* model)
{
    ModelImpl* m = reinterpret_cast<ModelImpl*>(model);
    RETURN_IF_TRITONRKNN_ERROR(m->Run());
    return nullptr;
}

int TRITONRKNN_ModelInputTensorNames(TRITONRKNN_Model* model, std::vector<std::string>& input_names)
{
    ModelImpl* m = reinterpret_cast<ModelImpl*>(model);
    return m->GetEngineInputNames(input_names);
}

int TRITONRKNN_ModelOutputTensorNames(TRITONRKNN_Model* model, std::vector<std::string>& output_names)
{
    ModelImpl* m = reinterpret_cast<ModelImpl*>(model);
    return m->GetEngineOutputNames(output_names);
}

namespace triton
{
    namespace backend 
    {
        namespace rknn
        {

            using TRITONRKNNModelHandle = std::shared_ptr<TRITONRKNN_Model>;
            using TRITONRKNNTensorHandle = std::shared_ptr<TRITONRKNN_Tensor>;

            class ModelState : public BackendModel
            {
            public:
                static TRITONSERVER_Error* Create(TRITONBACKEND_Model* triton_model, ModelState** state);
                virtual ~ModelState() = default;
                TRITONRKNN_Config* RknnConfig() { return &rknn_config_; }

            private:
                ModelState(TRITONBACKEND_Model* triton_model);

                // Auto-complete the model configuration
                TRITONSERVER_Error* AutoCompleteConfig();

                // Auto-complete the model IO
                TRITONSERVER_Error* AutoCompleteIO(const char* key, const std::vector<rknn_tensor_attr>& io_infos);

                // Auto-complete the model max batch size
                TRITONSERVER_Error* AutoCompleteMaxBatch(const std::vector<rknn_tensor_attr>& input_tensor_infos, const std::vector<rknn_tensor_attr>& output_tensor_infos);                

                // Validate that model configuration is supported by this backend
                TRITONSERVER_Error* ValidateModelConfig();

                // parse model config paramerters
                TRITONSERVER_Error* ParseBoolParameter(triton::common::TritonJson::Value& params, const std::string& mkey, bool* value);
                TRITONSERVER_Error* ParseIntParameter(triton::common::TritonJson::Value& params, const std::string& mkey, int* value);
                TRITONSERVER_Error* ParseParameters();

                TRITONRKNN_Config rknn_config_;
            };

            TRITONSERVER_Error* ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
            {
                try
                {
                    *state = new ModelState(triton_model);
                }
                catch (const BackendModelException& ex)
                {
                    RETURN_ERROR_IF_TRUE(ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL, std::string("unexpected nullptr in BackendModelException"));
                    RETURN_IF_ERROR(ex.err_);
                }

                // Auto-complete the configuration if requested...
                bool auto_complete_config = false;
                RETURN_IF_ERROR(TRITONBACKEND_ModelAutoCompleteConfig(triton_model, &auto_complete_config));
                if (auto_complete_config)
                {
                    RETURN_IF_ERROR((*state)->AutoCompleteConfig());

                    triton::common::TritonJson::WriteBuffer json_buffer;
                    (*state)->ModelConfig().Write(&json_buffer);

                    TRITONSERVER_Message* message;
                    RETURN_IF_ERROR(TRITONSERVER_MessageNewFromSerializedJson(&message, json_buffer.Base(), json_buffer.Size()));
                    RETURN_IF_ERROR(TRITONBACKEND_ModelSetConfig(triton_model, 1 /* config_version */, message));
                }

                RETURN_IF_ERROR((*state)->ValidateModelConfig());
                RETURN_IF_ERROR((*state)->ParseParameters());

                return nullptr;  // success
            }

            TRITONSERVER_Error* ModelState::ParseBoolParameter(triton::common::TritonJson::Value& params, const std::string& mkey, bool* value)
            {
                std::string value_str;
                RETURN_IF_ERROR(GetParameterValue(params, mkey, &value_str));
                RETURN_IF_ERROR(ParseBoolValue(value_str, value));

                return nullptr;
            }

            TRITONSERVER_Error* ModelState::ParseIntParameter(triton::common::TritonJson::Value& params, const std::string& mkey, int* value)
            {
                std::string value_str;
                RETURN_IF_ERROR(GetParameterValue(params, mkey, &value_str));
                RETURN_IF_ERROR(ParseIntValue(value_str, value));

                return nullptr;
            }

            TRITONSERVER_Error* ModelState::ParseParameters()
            {
                triton::common::TritonJson::Value params;
                bool status = ModelConfig().Find("parameters", &params);
                if (status)
                {
                    // want_float
                    bool enable_want_float = true;
                    auto err = ParseBoolParameter(params, "want_float", &enable_want_float);
                    if (err != nullptr)
                    {
                        if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND)
                        {
                            return err;
                        }
                        else
                        {
                            TRITONSERVER_ErrorDelete(err);
                        }
                    }
                    rknn_config_.want_float = (enable_want_float == true) ? 1 : 0;
                    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("want float is ") +
                        (enable_want_float ? "enabled" : "disabled") + " for model instance '" + Name() + "'").c_str());

                    // pass_through
                    bool enable_pass_through = false;
                    err = ParseBoolParameter(params, "pass_through", &enable_pass_through);
                    if (err != nullptr)
                    {
                        if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND)
                        {
                            return err;
                        }
                        else
                        {
                            TRITONSERVER_ErrorDelete(err);
                        }
                    }
                    rknn_config_.pass_through = (enable_pass_through == true) ? 1 : 0;
                    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("pass through is ") +
                        (enable_pass_through ? "enabled" : "disabled") + " for model instance '" + Name() + "'").c_str());

                    // flag
                    int flag = 0;
                    err = ParseIntParameter(params, "flag", &flag);
                    if (err != nullptr)
                    {
                        if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND)
                        {
                            return err;
                        }
                        else
                        {
                            TRITONSERVER_ErrorDelete(err);
                        }
                    }
                    rknn_config_.flag = flag;
                    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("flag is ") +
                        std::to_string(flag) + " for model instance '" + Name() + "'").c_str());
                }

                return nullptr;
            }

            ModelState::ModelState(TRITONBACKEND_Model* triton_model) : BackendModel(triton_model)
            {
            }

            TRITONSERVER_Error* ModelState::AutoCompleteMaxBatch(const std::vector<rknn_tensor_attr>& input_tensor_infos, const std::vector<rknn_tensor_attr>& output_tensor_infos)
            {
                // Determine if the model can potentially support batching. All
                // input and output tensors must have a variable first dimension.
                bool can_support_batching = true;
                for (const auto& io_info : input_tensor_infos)
                {
                    if ((io_info.n_dims == 0) || (io_info.dims[0] > 1))
                    {
                        can_support_batching = false;
                    }
                }
                for (const auto& io_info : output_tensor_infos)
                {
                    if ((io_info.n_dims == 0) || (io_info.dims[0] > 1))
                    {
                        can_support_batching = false;
                    }
                }

                // Set max-batch-size to 1 if we have determined that batching is
                // supported and max-batch-size is not specified. We need to update
                // the configuration itself as well as the cached value we have already
                // initialized in the model state.
                if (can_support_batching)
                {
                    if (MaxBatchSize() == 0) 
                    {
                        int default_max_batch_size = 1;
                        int max_batch_size = default_max_batch_size;

                        triton::common::TritonJson::Value mbs_value;
                        ModelConfig().Find("max_batch_size", &mbs_value);
                        mbs_value.SetInt(max_batch_size);
                        SetMaxBatchSize(max_batch_size);

                        LOG_MESSAGE(TRITONSERVER_LOG_WARN,
                            (std::string("autofilled max_batch_size to " + std::to_string(max_batch_size) + " for model '") +
                            Name() +  "' since batching is supporrted but no max_batch_size is "
                            "specified in model configuration. Must specify max_batch_size to utilize "
                            "autofill with a larger max batch size").c_str());
                    }

                    // Check to see if we need to turn on dynamic batching
                    // since model supports batching
                    if (MaxBatchSize() > 1)
                    {
                        triton::common::TritonJson::Value value;
                        bool found_sequence_batching = ModelConfig().Find("sequence_batching", &value);
                        bool found_dynamic_batching = ModelConfig().Find("dynamic_batching", &value);
                        if (!found_sequence_batching && !found_dynamic_batching)
                        {
                            triton::common::TritonJson::Value dynamic_batching(ModelConfig(), triton::common::TritonJson::ValueType::OBJECT);
                            triton::common::TritonJson::Value prefer_batch_size(ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
                            prefer_batch_size.AppendInt(MaxBatchSize());
                            dynamic_batching.Add("preferred_batch_size", std::move(prefer_batch_size));
                            RETURN_IF_ERROR(ModelConfig().Add("dynamic_batching", std::move(dynamic_batching)));
                        }
                    }
                }
                else if (MaxBatchSize() != 0)
                {
                    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG,
                        (std::string("autofill failed for model '") + Name() + "': model does not support batching while non-zero max_batch_size is specified").c_str());
                }

                return nullptr;  // success
            }

            TRITONSERVER_Error* ModelState::AutoCompleteIO(const char* key, const std::vector<rknn_tensor_attr>& io_infos)
            {
                triton::common::TritonJson::Value existing_ios;
                bool found_ios = ModelConfig().Find(key, &existing_ios);

                triton::common::TritonJson::Value ios(ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
                for (const auto& io_info : io_infos)
                {
                    triton::common::TritonJson::Value io_json(ModelConfig(), triton::common::TritonJson::ValueType::OBJECT);
                    std::string tensor_name = io_info.name;
                    RETURN_IF_ERROR(io_json.AddString("name", tensor_name));
                    RETURN_IF_ERROR(io_json.AddString("data_type", RknnDataTypeToModelConfigDataType(io_info.type)));

                    // The model signature supports batching then the first dimension
                    // is -1 and should not appear in the model configuration 'dims'
                    // that we are creating.
                    triton::common::TritonJson::Value dims(ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
                    for (size_t i = (MaxBatchSize() > 0) ? 1 : 0; i < io_info.n_dims; ++i)
                    {
                        RETURN_IF_ERROR(dims.AppendInt(io_info.dims[i]));
                    }

                    // If dims are empty then must use a reshape...
                    if (dims.ArraySize() == 0)
                    {
                        RETURN_IF_ERROR(dims.AppendInt(1));
                        triton::common::TritonJson::Value reshape(ModelConfig(), triton::common::TritonJson::ValueType::OBJECT);
                        triton::common::TritonJson::Value reshape_dims(ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
                        RETURN_IF_ERROR(reshape.Add("shape", std::move(reshape_dims)));
                        RETURN_IF_ERROR(io_json.Add("reshape", std::move(reshape)));
                    }
                    RETURN_IF_ERROR(io_json.Add("dims", std::move(dims)));
                    RETURN_IF_ERROR(ios.Append(std::move(io_json)));
                }

                if (found_ios)
                {
                    existing_ios.Swap(ios);
                }
                else
                {
                    ModelConfig().Add(key, std::move(ios));
                }

                return nullptr;  // success
            }

            TRITONSERVER_Error* ModelState::AutoCompleteConfig()
            {
                // Auto-complete configuration if requests
                // If the model configuration already specifies inputs and outputs
                // then don't perform any auto-completion.
                size_t input_cnt = 0;
                size_t output_cnt = 0;
                {
                    triton::common::TritonJson::Value inputs;
                    if (ModelConfig().Find("input", &inputs))
                    {
                        input_cnt = inputs.ArraySize();
                    }

                    triton::common::TritonJson::Value config_batch_inputs;
                    if (ModelConfig().Find("batch_input", &config_batch_inputs))
                    {
                        input_cnt += config_batch_inputs.ArraySize();
                    }

                    triton::common::TritonJson::Value outputs;
                    if (ModelConfig().Find("output", &outputs))
                    {
                        output_cnt = outputs.ArraySize();
                    }
                }

                if ((input_cnt > 0) && (output_cnt > 0))
                {
                    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("skipping model configuration auto-complete for '") +
                        Name() + "': inputs and outputs already specified").c_str());
                    return nullptr;  // success
                }

                std::string artifact_name;
                RETURN_IF_ERROR(ModelConfig().MemberAsString("default_model_filename", &artifact_name));
                if (artifact_name.empty())
                {
                    artifact_name = "model.rknn";
                }
                std::string model_path = JoinPath({RepositoryPath(), std::to_string(Version()), artifact_name});
                TRITONRKNN_Config* config = RknnConfig();
                std::shared_ptr<ModelImpl> model(new ModelImpl(artifact_name.c_str(), model_path.c_str(), config, -1));

                std::vector<rknn_tensor_attr> input_tensor_infos;
                std::vector<rknn_tensor_attr> output_tensor_infos;
                int ret = model->GetEngineTensorInfos(input_tensor_infos, output_tensor_infos);
                RETURN_ERROR_IF_TRUE(0 != ret, TRITONSERVER_ERROR_INTERNAL, std::string("rknpu engine get tensors info fail!, ret = ") + std::to_string(ret));

                RETURN_IF_ERROR(AutoCompleteMaxBatch(input_tensor_infos, output_tensor_infos));
                if (input_cnt == 0) 
                {
                    RETURN_IF_ERROR(AutoCompleteIO("input", input_tensor_infos));
                }
                if (output_cnt == 0)
                {
                    RETURN_IF_ERROR(AutoCompleteIO("output", output_tensor_infos));
                }

                return nullptr;  // success
            }

            TRITONSERVER_Error* ModelState::ValidateModelConfig()
            {
                triton::common::TritonJson::WriteBuffer buffer;
                RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
                LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("model configuration:\n") + buffer.Contents()).c_str());

                triton::common::TritonJson::Value ios;
                RETURN_IF_ERROR(ModelConfig().MemberAsArray("input", &ios));
                for (size_t i = 0; i < ios.ArraySize(); i++)
                {
                    triton::common::TritonJson::Value io;
                    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
                    std::string io_name;
                    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
                    // Check datatypes
                    std::string io_dtype;
                    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
                    RETURN_ERROR_IF_TRUE(ConvertDataType(io_dtype) == TRITONRKNN_DataType::TRITONRKNN_TYPE_INVALID,
                        TRITONSERVER_ERROR_INVALID_ARG, std::string("unsupported datatype '") + io_dtype + "' for tensor '" +
                            io_name + "' for model '" + Name() + "'");
                }
                RETURN_IF_ERROR(ModelConfig().MemberAsArray("output", &ios));
                for (size_t i = 0; i < ios.ArraySize(); i++)
                {
                    triton::common::TritonJson::Value io;
                    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
                    std::string io_name;
                    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
                    // Check datatypes
                    std::string io_dtype;
                    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
                    RETURN_ERROR_IF_TRUE(ConvertDataType(io_dtype) == TRITONRKNN_DataType::TRITONRKNN_TYPE_INVALID,
                        TRITONSERVER_ERROR_INVALID_ARG, std::string("unsupported datatype '") + io_dtype + "' for tensor '" +
                            io_name + "' for model '" + Name() + "'");
                }

                return nullptr;  // success
            }

            class ModelInstanceState : public BackendModelInstance
            {
            public:
                static TRITONSERVER_Error* Create(ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance, ModelInstanceState** state);
                virtual ~ModelInstanceState() = default;

                // Get the state of the model that corresponds to this instance.
                ModelState* StateForModel() const { return model_state_; }

                void ProcessRequests(TRITONBACKEND_Request** requests, const uint32_t request_count);

            private:
                ModelInstanceState(ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance);

                TRITONSERVER_Error* DetermineModelPath(const std::string& model_dir, std::string* model_path);

                TRITONSERVER_Error* SetInputTensors(size_t total_batch_size, TRITONBACKEND_Request** requests, const uint32_t request_count,
                    std::vector<TRITONBACKEND_Response*>* responses, std::map<std::string, TRITONRKNNTensorHandle>& input_tensors);

                TRITONSERVER_Error* ReadOutputTensors(size_t total_batch_size, const std::vector<std::string>& output_names,
                    TRITONBACKEND_Request** requests, const uint32_t request_count, std::vector<TRITONBACKEND_Response*>* responses, 
                    std::map<std::string, TRITONRKNNTensorHandle>& output_tensors);

                ModelState* model_state_;
                TRITONRKNNModelHandle triton_rknn_model_;
            };

            TRITONSERVER_Error* ModelInstanceState::Create(ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
                ModelInstanceState** state)
            {
                try
                {
                    *state = new ModelInstanceState(model_state, triton_model_instance);
                }
                catch (const BackendModelInstanceException& ex) {
                    RETURN_ERROR_IF_TRUE(ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
                        std::string("unexpected nullptr in BackendModelInstanceException"));
                    RETURN_IF_ERROR(ex.err_);
                }

                return nullptr;  // success
            }

            TRITONSERVER_Error* ModelInstanceState::DetermineModelPath(const std::string& model_dir, std::string* model_path)
            {
                bool exists;
                *model_path = JoinPath({model_dir, "model.rknn"});
                RETURN_IF_ERROR(FileExists(*model_path, &exists));
                if (not exists)
                {
                    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_NOT_FOUND,
                        std::string("rknn model should be named as 'model.rknn'").c_str());
                }

                return nullptr;
            }

            ModelInstanceState::ModelInstanceState(ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
                : BackendModelInstance(model_state, triton_model_instance), model_state_(model_state)
            {
                auto config = model_state->RknnConfig();
                int device_id = DeviceId();
                auto model_dir = JoinPath({model_state->RepositoryPath(), std::to_string(model_state->Version())});

                std::string model_path;
                THROW_IF_BACKEND_INSTANCE_ERROR(DetermineModelPath(model_dir, &model_path));

                TRITONRKNN_Model* triton_rknn_model = nullptr;
                auto instance_name = Name();
                THROW_IF_BACKEND_INSTANCE_ERROR(TRITONRKNN_ModelCreate(&triton_rknn_model, instance_name.c_str(), model_path.c_str(), config, device_id));
                triton_rknn_model_.reset(triton_rknn_model, TRITONRKNN_ModelDelete);
            }

            TRITONSERVER_Error* ModelInstanceState::SetInputTensors(size_t total_batch_size, TRITONBACKEND_Request** requests,
                const uint32_t request_count, std::vector<TRITONBACKEND_Response*>* responses, std::map<std::string, TRITONRKNNTensorHandle>& input_tensors)
            {
                BackendInputCollector collector(requests, request_count, responses,
                    StateForModel()->TritonMemoryManager(), StateForModel()->EnablePinnedInput(), CudaStream());

                const int max_batch_size = model_state_->MaxBatchSize();

                // All requests must have equally-sized input tensors so use any
                // request as the representative for the input tensors.
                uint32_t input_count;
                RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(requests[0], &input_count));

                for (uint32_t input_idx = 0; input_idx < input_count; ++input_idx)
                {
                    TRITONBACKEND_Input* input;
                    RETURN_IF_ERROR(TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input));

                    const char* name;
                    TRITONSERVER_DataType datatype;
                    const int64_t* shape;
                    uint32_t dims_count;
                    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(input, &name, &datatype, &shape, &dims_count, nullptr, nullptr));

                    // The shape for the entire input patch, [total_batch_size, ...]
                    std::vector<int64_t> batchn_shape(shape, shape + dims_count);

                    if (max_batch_size != 0)
                    {
                        batchn_shape[0] = total_batch_size;
                    }

                    TRITONRKNN_Tensor* tensor = nullptr;
                    TRITONRKNN_TensorCreate(&tensor, name, datatype, batchn_shape);
                    if (nullptr == tensor)
                    {
                        auto err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                            (std::string("Failed to create input tensor '") + name + "' with shape " + backend::ShapeToString(batchn_shape) +
                            " and data type " + TRITONSERVER_DataTypeString(datatype) + " for '" + Name() + "'").c_str());
                        return err;
                    }
                    input_tensors[name].reset(tensor, TRITONRKNN_TensorDelete);
                    collector.ProcessTensor(name, TRITONRKNN_TensorData(tensor), TRITONRKNN_TensorDataByteSize(tensor), TRITONSERVER_MEMORY_CPU, 0);
                }
                collector.Finalize();
                RETURN_IF_ERROR(TRITONRKNN_ModelSetInputTensors(triton_rknn_model_.get(), input_tensors));
                return nullptr;
            }

            TRITONSERVER_Error* ModelInstanceState::ReadOutputTensors(size_t total_batch_size, const std::vector<std::string>& output_names,
                TRITONBACKEND_Request** requests, const uint32_t request_count, std::vector<TRITONBACKEND_Response*>* responses, 
                std::map<std::string, TRITONRKNNTensorHandle>& output_tensors)
            {
                BackendOutputResponder responder(requests, request_count, responses, StateForModel()->MaxBatchSize(),
                    StateForModel()->TritonMemoryManager(), StateForModel()->EnablePinnedOutput(), CudaStream());

                RETURN_IF_ERROR(TRITONRKNN_ModelGetOutputTensors(triton_rknn_model_.get(), output_tensors));

                for (size_t idx = 0; idx < output_names.size(); ++idx)
                {
                    const std::string& name = output_names[idx];
                    TRITONRKNN_Tensor* tensor = nullptr;
                    if (output_tensors.end() != output_tensors.find(name))
                    {
                        tensor = output_tensors[name].get();
                    }
                    if (tensor == nullptr)
                    {
                        auto err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, 
                            (std::string("Failed to get output tensor '") + name + " for '" + Name() + "'").c_str());
                        return err;
                    }

                    auto dtype = ConvertDataType(TRITONRKNN_TensorDataType(tensor));
                    auto shape = TRITONRKNN_TensorShape(tensor).Shape();

                    responder.ProcessTensor(name, dtype, shape, TRITONRKNN_TensorData(tensor), TRITONSERVER_MEMORY_CPU, 0);
                }

                responder.Finalize();
                return nullptr;
            }

            void ModelInstanceState::ProcessRequests(TRITONBACKEND_Request** requests, const uint32_t request_count)
            {
                LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("TRITONBACKEND_ModelExecute: Running ") + Name() + " with " +
                    std::to_string(request_count) + " requests").c_str());

                uint64_t exec_start_ns = 0;
                SET_TIMESTAMP(exec_start_ns);

                const int max_batch_size = model_state_->MaxBatchSize();
                // For each request collect the total batch size for this inference
                // execution. The batch-size, number of inputs, and size of each
                // input has already been checked so don't need to do that here.
                size_t total_batch_size = 0;
                for (size_t i = 0; i < request_count; ++i)
                {
                    // If we get a nullptr request then something is badly wrong. Fail
                    // and release all requests.
                    if (requests[i] == nullptr)
                    {
                        RequestsRespondWithError(requests, request_count, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, 
                            std::string("null request given to rknn backend for '" + Name() + "'").c_str()));
                        return;
                    }

                    if (max_batch_size > 0)
                    {
                        // Retrieve the batch size from one of the inputs, if the model
                        // supports batching, the first dimension size is batch size
                        TRITONBACKEND_Input* input;
                        TRITONSERVER_Error* err = TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
                        if (err == nullptr)
                        {
                            const int64_t* shape;
                            err = TRITONBACKEND_InputProperties(input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
                            total_batch_size += shape[0];
                        }
                        else
                        {
                            RequestsRespondWithError(requests, request_count, err);
                            return;
                        }
                    }
                    else
                    {
                        total_batch_size += 1;
                    }
                }

                // If there are no valid requests then no need to run the
                // inference. This should never happen unless called with an empty
                // 'requests' for some reason.
                if (total_batch_size == 0)
                {
                    return;
                }

                // Make sure the maximum batch size is not exceeded. The
                // total_batch_size must be 1 for models that don't support batching
                // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
                // scheduler has done something badly wrong so fail and release all
                // requests
                if ((total_batch_size != 1) and (total_batch_size > static_cast<size_t>(max_batch_size)))
                {
                    RequestsRespondWithError(requests, request_count, TRITONSERVER_ErrorNew(
                            TRITONSERVER_ERROR_INTERNAL, std::string("batch size " + std::to_string(total_batch_size) + " for '" +
                                Name() + "', max allowed is " + std::to_string(max_batch_size)).c_str()));
                    return;
                }

                // At this point we are committed to running inference with all
                // 'requests'. Create a response for each request. During input
                // processing if there is an error with any request that error will
                // be sent immediately with the corresponding response (and the
                // response pointer will then be nullptr). The request object
                // itself will not be released until after all inferencing is done
                // (below) as we may need to access the request object when
                // determine how to process outputs (for example, even if we don't
                // need the outputs for a request that has an error,  we do need to
                // know the size of those outputs associated with the request so we
                // can skip them in the output tensors).
                std::vector<TRITONBACKEND_Response*> responses;
                responses.reserve(request_count);

                for (size_t i = 0; i < request_count; ++i)
                {
                    TRITONBACKEND_Response* response;
                    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
                    if (err == nullptr)
                    {
                        responses.emplace_back(response);
                    }
                    else
                    {
                        responses.emplace_back(nullptr);
                        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
                        TRITONSERVER_ErrorDelete(err);
                    }
                }

                // Collect the names of requested outputs. Do not include outputs
                // for requests that have already responded with an error.
                // TODO: understand here
                std::vector<std::string> required_outputs;
                int ret = TRITONRKNN_ModelOutputTensorNames(triton_rknn_model_.get(), required_outputs);
                if (0 != ret)
                {
                    auto err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, 
                        (std::string("Failed to get output tensor names for '") + Name() + "'").c_str());
                    SendErrorForResponses(&responses, request_count, err);
                    return;
                }

                bool all_response_failed = false;

                std::map<std::string, TRITONRKNNTensorHandle> input_tensors;
                RESPOND_ALL_AND_SET_TRUE_IF_ERROR(responses, request_count, all_response_failed,
                    SetInputTensors(total_batch_size, requests, request_count, &responses, input_tensors));

                uint64_t compute_start_ns = 0;
                SET_TIMESTAMP(compute_start_ns);

                if (!all_response_failed)
                {
                    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(responses, request_count, all_response_failed,
                        TRITONRKNN_ModelRun(triton_rknn_model_.get()));
                }

                uint64_t compute_end_ns = 0;
                SET_TIMESTAMP(compute_end_ns);

                std::map<std::string, TRITONRKNNTensorHandle> output_tensors;
                if (!all_response_failed)
                {
                    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(responses, request_count, all_response_failed,
                        ReadOutputTensors(total_batch_size, required_outputs, requests, request_count, &responses, output_tensors));
                }

                uint64_t exec_end_ns = 0;
                SET_TIMESTAMP(exec_end_ns);

                // Send all the responses that haven't already been sent because of
                // an earlier error. Note that the responses are not set to nullptr
                // here as we need that indication below to determine if the request
                // we successful or not.
                for (auto& response : responses)
                {
                    if (response != nullptr)
                    {
                        LOG_IF_ERROR(TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
                            "failed to send rknn backend response");
                    }
                }

                // Report statistics for each request.
                for (uint32_t r = 0; r < request_count; ++r)
                {
                    auto& request = requests[r];
                    LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportStatistics(TritonModelInstance(), request,
                        (responses[r] != nullptr) /* success */, exec_start_ns, compute_start_ns, compute_end_ns, exec_end_ns),
                        "failed reporting request statistics");

                    LOG_IF_ERROR(TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL), "failed releasing request");
                }

                if (!all_response_failed)
                {
                    // TODO: Report the entire batch statistics.
                    LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportBatchStatistics(TritonModelInstance(), total_batch_size, exec_start_ns,
                        compute_start_ns, compute_end_ns, exec_end_ns), "failed reporting batch request statistics");
                }
                LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("TRITONBACKEND_ModelExecute: model ") + Name() +
                    " released " + std::to_string(request_count) + " requests").c_str());
            }

            extern "C" 
            {
                TRITONSERVER_Error* TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
                {
                    const char* cname;
                    RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
                    std::string name(cname);

                    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

                    uint32_t api_version_major, api_version_minor;
                    RETURN_IF_ERROR(TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

                    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("Triton TRITONBACKEND API version: ") +
                        std::to_string(api_version_major) + "." + std::to_string(api_version_minor)).c_str());

                    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("'") + name + "' TRITONBACKEND API version: " +
                        std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." + std::to_string(TRITONBACKEND_API_VERSION_MINOR)).c_str());

                    if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
                        (api_version_minor < TRITONBACKEND_API_VERSION_MINOR))
                    {
                        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED,
                            (std::string("Triton TRITONBACKEND API version: ") + std::to_string(api_version_major) + "." +
                            std::to_string(api_version_minor) + " does not support '" + name + "' TRITONBACKEND API version: " +
                            std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." + std::to_string(TRITONBACKEND_API_VERSION_MINOR)).c_str());
                    }

                    // The backend configuration may contain information needed by the
                    // backend, such a command-line arguments.
                    TRITONSERVER_Message* backend_config_message;
                    RETURN_IF_ERROR(TRITONBACKEND_BackendConfig(backend, &backend_config_message));

                    const char* buffer;
                    size_t byte_size;
                    RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(backend_config_message, &buffer, &byte_size));
                    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("backend configuration:\n") + buffer).c_str());

                    return nullptr;  // success
                }

                TRITONSERVER_Error* TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
                {
                    const char* cname;
                    RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
                    std::string name(cname);

                    uint64_t version;
                    RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

                    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
                        (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " + std::to_string(version) + ")").c_str());

                    // Create a ModelState object and associate it with the
                    // TRITONBACKEND_Model.
                    ModelState* model_state = nullptr;
                    RETURN_IF_ERROR(ModelState::Create(model, &model_state));
                    RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

                    return nullptr;  // success
                }

                TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
                {
                    void* vstate;
                    RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
                    ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

                    LOG_MESSAGE(TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");
                    delete model_state;

                    return nullptr;  // success
                }

                TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
                {
                    const char* cname;
                    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
                    std::string name(cname);

                    int32_t device_id;
                    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
                    TRITONSERVER_InstanceGroupKind kind;
                    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

                    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
                        TRITONSERVER_InstanceGroupKindString(kind) + " device " + std::to_string(device_id) + ")").c_str());

                    // Get the model state associated with this instance's model
                    TRITONBACKEND_Model* model;
                    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

                    void* vmodelstate;
                    RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
                    ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

                    // With each instance we create a ModelInstanceState object and
                    // associate it with the TRITONBACKEND_ModelInstance.
                    ModelInstanceState* instance_state;
                    RETURN_IF_ERROR(ModelInstanceState::Create(model_state, instance, &instance_state));
                    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(instance, reinterpret_cast<void*>(instance_state)));

                    return nullptr;
                }

                TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
                {
                    void* vstate;
                    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
                    ModelInstanceState* instance_state = reinterpret_cast<ModelInstanceState*>(vstate);

                    LOG_MESSAGE(TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelInstanceFinalize: delete instance state");
                    delete instance_state;

                    return nullptr;
                }

                TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests, const uint32_t request_count)
                {
                    // Triton will not call this function simultaneously for the same
                    // 'instance'. But since this backend could be used by multiple
                    // instances from multiple models the implementation needs to handle
                    // multiple calls to this function at the same time (with different
                    // 'instance' objects). Suggested practice for this is to use only
                    // function-local and model-instance-specific state (obtained from
                    // 'instance'), which is what we do here.
                    ModelInstanceState* instance_state;
                    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, reinterpret_cast<void**>(&instance_state)));
                    ModelState* model_state = instance_state->StateForModel();

                    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("model ") + model_state->Name() + ", instance " +
                        instance_state->Name() + ", executing " + std::to_string(request_count) + " requests").c_str());

                    // At this point we accept ownership of 'requests', which means that
                    // even if something goes wrong we must still return success from
                    // this function. If something does go wrong in processing a
                    // particular request then we send an error response just for the
                    // specific request.
                    instance_state->ProcessRequests(requests, request_count);

                    return nullptr;  // success
                }

            }  // extern "C"
        }
    }
}  // namespace triton::backend::rknn
