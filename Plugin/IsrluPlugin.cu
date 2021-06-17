/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cassert>
#include <cstring>
#include <vector>
#include <cmath>

#include "NvInfer.h"
#include "IsrluPlugin.h"
#include "bertCommon.h"
#include "common.h"
#include "serialize.hpp"

using namespace nvinfer1;

namespace bert
{
    constexpr float alpha = 3;  // fixing the value of the hyper-parameter.

template <typename T, unsigned TPB>  // TPB : Threads Per Block.
__global__ void IsrluKernel(const T alpha, int n, const T* input, T* output)
{

    const int idx = blockIdx.x * TPB + threadIdx.x;

    if (idx < n)
    {
        const T in = input[idx];
        
        if(in >= 0)
        {
            const T temp = in;
        }
        else
        {
            const T temp = in * (1 / sqrt(1 + alpha * in * in));
        }
        
        output[idx] = temp;
    }
}

inline int computeIsrlu(cudaStream_t stream, int n, const float* input, float* output)
{

    constexpr int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    <IsrluKernel<float, blockSize><<<gridSize, blockSize, 0, stream>>>(alpha, n, input, output);

    CHECK(cudaPeekAtLastError());
    return 0;
}

inline int computeIsrlu(cudaStream_t stream, int n, const half* input, half* output)
{
    const int blockSize = 256;

    if (0 == (n & 1))
    {
        const int n2 = n / 2;       // for enabling half precision (FP16)
        const int gridSize = (n2 + blockSize - 1) / blockSize;
        
        const half2 alpha2 = __floats2half2_rn(alpha, aplha);
        
        const half2* input2 = reinterpret_cast<const half2*>(input);
        half2* output2 = reinterpret_cast<half2*>(output);
        
        IsrluKernel<half2, blockSize><<<gridSize, blockSize, 0, stream>>>(alpha2, n2, input2, output2);
    }
    else
    {
        const int gridSize = (n + blockSize - 1) / blockSize;
        IsrluKernel<half, blockSize><<<gridSize, blockSize, 0, stream>>>(alpha, n, input, output);
    }

    CHECK(cudaPeekAtLastError());
    return 0;
}

template <typename T, int TPB>  
__global__ void IsrluBiasKernel(const T alpha, T* output, const T* input, const T* bias, const int ld)
{

    const int offset = blockIdx.x * ld;

    for (int it = threadIdx.x; it < ld; it += TPB)
    {
        const int idx = it + offset;
        const T in = input[idx] + bias[it];       
        if(in >= 0)
        {
            const T temp = in;
        }
        else
        {
            const T temp = in * (1 / sqrt(1 + alpha * in * in));
        }
        
        output[idx] = temp;
         }
        
    }
}

void computeIsrluBias(
    float* output, const float* input, const float* bias, const int ld, const int cols, cudaStream_t stream)
{
    IsrluBiasKernel<float, 256><<<cols, 256, 0, stream>>>(alpha, output, input, bias, ld);
    CHECK(cudaPeekAtLastError());
}

void computeIsrluBias(
    half* output, const half* input, const half* bias, const int ld, const int cols, cudaStream_t stream)
{
    if (ld & 1)
    {
        IsrluBiasKernel<half, 256><<<cols, 256, 0, stream>>>(alpha, output, input, bias, ld);
    }
    else
    {
        const half2 aplha2 = __floats2half2_rn(alpha, alpha);
        const int ld2 = ld / 2;
        
        const half2* input2 = reinterpret_cast<const half2*>(input);
        const half2* bias2 = reinterpret_cast<const half2*>(bias);
        half2* output2 = reinterpret_cast<half2*>(output);
        
        geluIsrluKernel<half2, 256><<<cols, 256, 0, stream>>>(alpha2, output2, input2, bias2, ld2);
    }

    CHECK(cudaPeekAtLastError());
}

/////////////////////////////////

namespace
{
static const char* ISRLU_PLUGIN_VERSION{"1"};
static const char* ISRLU_PLUGIN_NAME{"CustomIsrluPluginDynamic"};
} // namespace

// Static class fields initialization
PluginFieldCollection IsrluPluginDynamicCreator::mFC{};
std::vector<PluginField> IsrluPluginDynamicCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(IsrluPluginDynamicCreator);

IsrluPluginDynamic::IsrluPluginDynamic(const std::string name, const DataType type)
    : mLayerName(name)
    , mType(type)
    , mHasBias(false)
    , mLd(0)
{
    mBias.values = nullptr;
    mBias.count = 0;
}

IsrluPluginDynamic::IsrluPluginDynamic(const std::string name, const DataType type, const Weights B)
    : mLayerName(name)
    , mType(type)
    , mHasBias(true)
    , mBias(B)
    , mLd(B.count)
{
}

IsrluPluginDynamic::IsrluPluginDynamic(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    gLogVerbose << "Starting to deserialize ISRLU plugin" << std::endl;
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mLd);
    deserialize_value(&data, &length, &mHasBias);

    gLogVerbose << "Deserialized parameters: mLd: " << mLd << ", mHasBias: " << mHasBias << std::endl;
    if (mHasBias)
    {
        const char* d = static_cast<const char*>(data);
        gLogVerbose << "Deserializing Bias" << std::endl;
        if (mLd <= 0)
        {
            gLogError << "Isrlu + bias: deserialization inconsistent. HasBias but mLd is 0" << std::endl;
        }
        const size_t wordSize = samplesCommon::getElementSize(mType);
        mBiasDev = deserToDev<char>(d, mLd * wordSize);
    }
    gLogVerbose << "Finished deserializing ISRLU plugin" << std::endl;
    mBias.values = nullptr;
    mBias.count = mLd;
}
// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* IsrluPluginDynamic::clone() const
{
    if (mHasBias)
    {
        return new IsrluPluginDynamic(mLayerName, mType, mBias);
    }
    return new IsrluPluginDynamic(mLayerName, mType);
}

nvinfer1::DimsExprs IsrluPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{
    return inputs[0];
}

bool IsrluPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{

    const PluginTensorDesc& input = inOut[0];
    if (pos == 0)
    {
        return (input.type == mType) && (input.format == TensorFormat::kLINEAR);
    }
    if (pos == 1)
    {
        const PluginTensorDesc& output = inOut[1];
        return (input.type == output.type) && (output.format == TensorFormat::kLINEAR);
    }
    return false;
}

void IsrluPluginDynamic::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    assert(mType == in[0].desc.type);
}

size_t IsrluPluginDynamic::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const
{
    return 0;
}
int IsrluPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    const int inputVolume = volume(inputDesc[0].dims);

    int status = -1;

    // Our plugin outputs only one tensor
    // Launch CUDA kernel wrapper and save its return value
    if (mType == DataType::kFLOAT)
    {
        const float* input = static_cast<const float*>(inputs[0]);
        float* output = static_cast<float*>(outputs[0]);
        if (mHasBias)
        {
            const float* bias = reinterpret_cast<float*>(mBiasDev);
            const int cols = inputVolume / mLd;
            const int rows = mLd;
            computeIsrluBias(output, input, bias, rows, cols, stream);
        }
        else
        {
            status = computeIsrlu(stream, inputVolume, input, output);
        }
    }
    else if (mType == DataType::kHALF)
    {
        const half* input = static_cast<const half*>(inputs[0]);

        half* output = static_cast<half*>(outputs[0]);

        if (mHasBias)
        {
            const half* bias = reinterpret_cast<half*>(mBiasDev);
            const int cols = inputVolume / mLd;
            const int rows = mLd;
            computeIsrluBias(output, input, bias, rows, cols, stream);
        }
        else
        {
            status = computeIsrlu(stream, inputVolume, input, output);
        }
    }
    else
    {
        assert(false);
    }

    return status;
}

// IPluginV2Ext Methods
nvinfer1::DataType IsrluPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    assert(index == 0);
    assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
    return inputTypes[0];
}

// IPluginV2 Methods

const char* IsrluPluginDynamic::getPluginType() const
{
    return ISRLU_PLUGIN_NAME;
}

const char* IsrluPluginDynamic::getPluginVersion() const
{
    return ISRLU_PLUGIN_VERSION;
}

int IsrluPluginDynamic::getNbOutputs() const
{
    return 1;
}

int IsrluPluginDynamic::initialize()
{
    gLogVerbose << "ISRLU init start" << std::endl;
    if (mHasBias && mBias.values)
    {
        // target size
        const size_t wordSize = samplesCommon::getElementSize(mType);
        const size_t nbBytes = mBias.count * wordSize;
        CHECK(cudaMalloc(&mBiasDev, nbBytes));

        if (mType == DataType::kFLOAT)
        {
            convertAndCopyToDevice(mBias, reinterpret_cast<float*>(mBiasDev));
        }
        else
        {
            convertAndCopyToDevice(mBias, reinterpret_cast<half*>(mBiasDev));
        }
    }
    gLogVerbose << "ISRLU init done" << std::endl;
    return 0;
}

void IsrluPluginDynamic::terminate()
{
    if (mHasBias)
    {
        CHECK(cudaFree(mBiasDev));
    }
}

size_t IsrluPluginDynamic::getSerializationSize() const
{
    const size_t wordSize = samplesCommon::getElementSize(mType);
    const size_t biasSize = mHasBias ? mLd * wordSize : 0;
    return sizeof(mType) + sizeof(mHasBias) + sizeof(mLd) + biasSize;
}

void IsrluPluginDynamic::serialize(void* buffer) const
{
    serialize_value(&buffer, mType);
    serialize_value(&buffer, mLd);
    serialize_value(&buffer, mHasBias);
    if (mHasBias)
    {
        char *d = static_cast<char*>(buffer);
        const size_t wordSize = samplesCommon::getElementSize(mType);
        const size_t biasSize = mHasBias ? mLd * wordSize : 0;
        if (biasSize <= 0)
        {
            gLogError << "Isrlu + bias: bias size inconsistent" << std::endl;
        }
        serFromDev(d, mBiasDev, mLd * wordSize);
    }
}

void IsrluPluginDynamic::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void IsrluPluginDynamic::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* IsrluPluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

///////////////

IsrluPluginDynamicCreator::IsrluPluginDynamicCreator()
{

    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* IsrluPluginDynamicCreator::getPluginName() const
{
    return ISRLU_PLUGIN_NAME;
}

const char* IsrluPluginDynamicCreator::getPluginVersion() const
{
    return ISRLU_PLUGIN_VERSION;
}

const PluginFieldCollection* IsrluPluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* IsrluPluginDynamicCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{

    Weights bias{DataType::kFLOAT, nullptr, 0};
    int typeId = -1;
    for (int i = 0; i < fc->nbFields; i++)
    {
        std::string field_name(fc->fields[i].name);

        if (field_name.compare("type_id") == 0)
        {
            typeId = *static_cast<const int*>(fc->fields[i].data);
            gLogVerbose << "Building typeId: " << typeId << std::endl;
        }

        if (field_name.compare("bias") == 0)
        {
            gLogVerbose << "Building bias...\n";
            bias.values = fc->fields[i].data;
            bias.count = fc->fields[i].length;
            bias.type = fieldTypeToDataType(fc->fields[i].type);
        }
    }

    if (typeId < 0 || typeId > 3)
    {
        gLogError << "ISRLU: invalid typeId " << typeId << std::endl;
        return nullptr;
    }
    DataType type = static_cast<DataType>(typeId);
    gLogVerbose << "Creating IsrluPluginDynamic...\n";
    if (bias.values == nullptr)
    {
        return new IsrluPluginDynamic(name, type);
    }

    return new IsrluPluginDynamic(name, type, bias);
}

IPluginV2* IsrluPluginDynamicCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call GeluPluginDynamic::destroy()
    return new IsrluPluginDynamic(name, serialData, serialLength);
}

void IsrluPluginDynamicCreator::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* IsrluPluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}
}
