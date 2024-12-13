// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "milvus/types/FieldData.h"

#include <type_traits>
#include <utility>
#include <vector>

#include "milvus/types/DataType.h"
#include "milvus/types/FloatUtils.h"

namespace milvus {

namespace {

template <DataType Dt>
struct DataTypeTraits {
    static const bool is_vector = false;
};

template <>
struct DataTypeTraits<DataType::BINARY_VECTOR> {
    static const bool is_vector = true;
};

template <>
struct DataTypeTraits<DataType::FLOAT_VECTOR> {
    static const bool is_vector = true;
};

template <typename T, DataType Dt, std::enable_if_t<!DataTypeTraits<Dt>::is_vector, bool> = true>
StatusCode
AddElement(const T& element, std::vector<T>& array) {
    array.push_back(element);
    return StatusCode::OK;
}

template <typename T, DataType Dt, std::enable_if_t<DataTypeTraits<Dt>::is_vector, bool> = true>
StatusCode
AddElement(const T& element, std::vector<T>& array) {
    if (element.empty()) {
        return StatusCode::VECTOR_IS_EMPTY;
    }

    if (!array.empty() && element.size() != array.at(0).size()) {
        return StatusCode::DIMENSION_NOT_EQUAL;
    }

    array.emplace_back(element);
    return StatusCode::OK;
}

}  // namespace

const std::string&
Field::Name() const {
    return name_;
}

DataType
Field::Type() const {
    return data_type_;
}

Field::Field(std::string name, DataType data_type) : name_(std::move(name)), data_type_(data_type) {
}

template <typename T, DataType Dt>
FieldData<T, Dt>::FieldData() : Field("", Dt) {
}

template <typename T, DataType Dt>
FieldData<T, Dt>::FieldData(std::string name) : Field(std::move(name), Dt) {
}

template <typename T, DataType Dt>
FieldData<T, Dt>::FieldData(std::string name, const std::vector<T>& data) : Field(std::move(name), Dt), data_{data} {
}

template <typename T, DataType Dt>
FieldData<T, Dt>::FieldData(std::string name, std::vector<T>&& data)
    : Field(std::move(name), Dt), data_{std::move(data)} {
}

template <typename T, DataType Dt>
StatusCode
FieldData<T, Dt>::Add(const T& element) {
    return AddElement<T, Dt>(element, data_);
}

template <typename T, DataType Dt>
StatusCode
FieldData<T, Dt>::Add(T&& element) {
    return AddElement<T, Dt>(std::move(element), data_);
}

template <typename T, DataType Dt>
size_t
FieldData<T, Dt>::Count() const {
    return data_.size();
}

template <typename T, DataType Dt>
const std::vector<T>&
FieldData<T, Dt>::Data() const {
    return data_;
}

template <typename T, DataType Dt>
std::vector<T>&
FieldData<T, Dt>::Data() {
    return data_;
}

template <DataType Dt>
BinaryVecFieldDataImpl<Dt>::BinaryVecFieldDataImpl() : FieldData<std::string, Dt>() {
}

template <DataType Dt>
BinaryVecFieldDataImpl<Dt>::BinaryVecFieldDataImpl(std::string name) : FieldData<std::string, Dt>(std::move(name)) {
}

template <DataType Dt>
BinaryVecFieldDataImpl<Dt>::BinaryVecFieldDataImpl(std::string name, const std::vector<std::string>& data)
    : FieldData<std::string, Dt>(std::move(name), data) {
}

template <DataType Dt>
BinaryVecFieldDataImpl<Dt>::BinaryVecFieldDataImpl(std::string name, std::vector<std::string>&& data)
    : FieldData<std::string, Dt>(std::move(name), std::move(data)) {
}

template <DataType Dt>
BinaryVecFieldDataImpl<Dt>::BinaryVecFieldDataImpl(std::string name, const std::vector<std::vector<uint8_t>>& data)
    : FieldData<std::string, Dt>(std::move(name), CreateBinaryStrings(data)) {
}

template <DataType Dt>
std::vector<std::vector<uint8_t>>
BinaryVecFieldDataImpl<Dt>::DataAsUnsignedChars() const {
    std::vector<std::vector<uint8_t>> ret;
    ret.reserve(this->data_.size());
    for (const auto& item : this->data_) {
        ret.emplace_back(item.begin(), item.end());
    }
    return ret;
}

template <DataType Dt>
StatusCode
BinaryVecFieldDataImpl<Dt>::Add(const std::string& element) {
    return AddElement<std::string, Dt>(element, this->data_);
}

template <DataType Dt>
StatusCode
BinaryVecFieldDataImpl<Dt>::Add(std::string&& element) {
    return AddElement<std::string, Dt>(element, this->data_);
}

template <DataType Dt>
StatusCode
BinaryVecFieldDataImpl<Dt>::Add(const std::vector<uint8_t>& element) {
    return Add(std::string{element.begin(), element.end()});
}

template <DataType Dt>
std::vector<std::string>
BinaryVecFieldDataImpl<Dt>::CreateBinaryStrings(const std::vector<std::vector<uint8_t>>& data) {
    std::vector<std::string> ret;
    ret.reserve(data.size());
    for (const auto& item : data) {
        ret.emplace_back(item.begin(), item.end());
    }
    return ret;
}

template <DataType Dt>
std::string
BinaryVecFieldDataImpl<Dt>::CreateBinaryString(const std::vector<uint8_t>& data) {
    return std::string{data.begin(), data.end()};
}

template <typename Fp16T, DataType Dt>
Fp16VecFieldData<Fp16T, Dt>::Fp16VecFieldData(std::string name, const std::vector<std::vector<Fp16T>>& data)
    : Fp16VecFieldData<Fp16T, Dt>(std::move(name), CreateBinaryStringsFromFloats(data)) {
}

template <typename Fp16T, DataType Dt>
Fp16VecFieldData<Fp16T, Dt>::Fp16VecFieldData(std::string name, std::vector<std::vector<Fp16T>>&& data)
    : Fp16VecFieldData<Fp16T, Dt>(std::move(name), CreateBinaryStringsFromFloats(std::move(data))) {
}

template <typename Fp16T, DataType Dt>
Fp16VecFieldData<Fp16T, Dt>::Fp16VecFieldData(std::string name, const std::vector<std::vector<float>>& data)
    : Fp16VecFieldData<Fp16T, Dt>(std::move(name), CreateBinaryStringsFromFloats(data)) {
}

template <typename Fp16T, DataType Dt>
Fp16VecFieldData<Fp16T, Dt>::Fp16VecFieldData(std::string name, const std::vector<std::vector<double>>& data)
    : Fp16VecFieldData<Fp16T, Dt>(std::move(name), CreateBinaryStringsFromFloats(data)) {
}

template <typename Fp16T, DataType Dt>
template <typename T>
std::vector<std::vector<T>>
Fp16VecFieldData<Fp16T, Dt>::DataAsFloats() const {
    static_assert(std::is_same_v<Fp16T, Eigen::half> || std::is_same_v<Fp16T, Eigen::bfloat16>,
                  "Fp16T should be Eigen::half or Eigen::bfloat16");
    std::vector<std::vector<T>> result;
    result.reserve(this->data_.size());
    for (const typename Fp16VecFieldData<Fp16T, Dt>::ElementT& str : this->data_) {
        if constexpr (std::is_same_v<Fp16T, Eigen::half>) {
            std::vector<T> float_vec = CreateFloatVecFromFp16Bytes<Fp16T, T>(str);
            result.push_back(std::move(float_vec));
        }
    }
    return result;
}

template <typename Fp16T, typename FloatT>
struct FloatVecToBytesTrait;

template <>
struct FloatVecToBytesTrait<Eigen::half, Eigen::half> {
    static std::string
    Convert(const std::vector<Eigen::half>& data) {
        return Fp16VecToBytes(data);
    }
};

template <>
struct FloatVecToBytesTrait<Eigen::half, float> {
    static std::string
    Convert(const std::vector<float>& data) {
        return FloatToFp16Bytes(data);
    }
};

template <>
struct FloatVecToBytesTrait<Eigen::half, double> {
    static std::string
    Convert(const std::vector<double>& data) {
        return DoubleToFp16Bytes(data);
    }
};

template <>
struct FloatVecToBytesTrait<Eigen::bfloat16, Eigen::bfloat16> {
    static std::string
    Convert(const std::vector<Eigen::bfloat16>& data) {
        return Bf16VecToBytes(data);
    }
};

template <>
struct FloatVecToBytesTrait<Eigen::bfloat16, float> {
    static std::string
    Convert(const std::vector<float>& data) {
        return FloatToBf16Bytes(data);
    }
};

template <>
struct FloatVecToBytesTrait<Eigen::bfloat16, double> {
    static std::string
    Convert(const std::vector<double>& data) {
        return DoubleToBf16Bytes(data);
    }
};

template <typename Fp16T, DataType Dt>
template <typename FloatT>
std::vector<std::string>
Fp16VecFieldData<Fp16T, Dt>::CreateBinaryStringsFromFloats(const std::vector<std::vector<FloatT>>& data) {
    std::vector<std::string> result;
    result.reserve(data.size());
    for (const auto& item : data) {
        std::string bytes = FloatVecToBytesTrait<Fp16T, FloatT>::Convert(item);
        result.push_back(std::move(bytes));
    }
    return result;
}

// explicit declare FieldData
template class FieldData<bool, DataType::BOOL>;
template class FieldData<int8_t, DataType::INT8>;
template class FieldData<int16_t, DataType::INT16>;
template class FieldData<int32_t, DataType::INT32>;
template class FieldData<int64_t, DataType::INT64>;
template class FieldData<float, DataType::FLOAT>;
template class FieldData<double, DataType::DOUBLE>;
template class FieldData<std::string, DataType::VARCHAR>;
template class FieldData<std::string, DataType::BINARY_VECTOR>;
template class FieldData<std::vector<float>, DataType::FLOAT_VECTOR>;
template class Fp16VecFieldData<Eigen::bfloat16, DataType::BFLOAT16_VECTOR>;
template class Fp16VecFieldData<Eigen::half, DataType::FLOAT16_VECTOR>;

}  // namespace milvus
