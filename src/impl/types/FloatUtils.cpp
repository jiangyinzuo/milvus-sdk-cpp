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

#include "milvus/types/FloatUtils.h"

#include <cstdint>
#include <type_traits>

#include "Eigen/Core"

namespace milvus {

template <typename Fp16T, typename FloatT>
struct Fp16ToFloatTraits;

template <>
struct Fp16ToFloatTraits<Eigen::half, Eigen::half> {
    static inline Eigen::half
    convert(uint16_t val) {
        return Eigen::half_impl::raw_uint16_to_half(val);
    }
};

template <>
struct Fp16ToFloatTraits<Eigen::bfloat16, Eigen::bfloat16> {
    static inline Eigen::bfloat16
    convert(uint16_t val) {
        return Eigen::bfloat16_impl::raw_uint16_to_bfloat16(val);
    }
};

template <typename FloatT>
struct Fp16ToFloatTraits<Eigen::half, FloatT> {
    static_assert(std::is_same_v<FloatT, float> || std::is_same_v<FloatT, double>, "FloatT should be float or double");
    static FloatT
    convert(uint16_t val) {
        Eigen::half h = Eigen::half_impl::raw_uint16_to_half(val);
        return Eigen::half_impl::half_to_float(h);
    }
};

template <typename FloatT>
struct Fp16ToFloatTraits<Eigen::bfloat16, FloatT> {
    static_assert(std::is_same_v<FloatT, float> || std::is_same_v<FloatT, double>, "FloatT should be float or double");
    static FloatT
    convert(uint16_t val) {
        Eigen::bfloat16 b = Eigen::bfloat16_impl::raw_uint16_to_bfloat16(val);
        return Eigen::bfloat16_impl::bfloat16_to_float(b);
    }
};

template <typename Fp16T, typename T>
static std::vector<T>
Fp16BytesToFloatVecImpl(const std::string& val) {
    static_assert(sizeof(Fp16T) == 2, "Fp16T should be 2 bytes");
    std::vector<T> result;
    assert(val.size() % 2 == 0);
    result.reserve(val.size() / 2);
    for (size_t i = 0; i < val.size(); i += 2) {
        union {
            uint16_t u16;
            char bytes[2];
        } value;
        if constexpr (BYTE_ORDER == LITTLE_ENDIAN) {
            value.bytes[0] = val[i];
            value.bytes[1] = val[i + 1];
        } else {
            value.bytes[0] = val[i + 1];
            value.bytes[1] = val[i];
        }
        T fp = Fp16ToFloatTraits<Fp16T, T>::convert(value.u16);
        result.push_back(fp);
    }
    return result;
}

template <typename T>
std::vector<T>
Fp16BytesToFloatVec(const std::string& val) {
  return Fp16BytesToFloatVecImpl<Eigen::half, T>(val);
}

template std::vector<float>
Fp16BytesToFloatVec(const std::string& val);

template std::vector<double>
Fp16BytesToFloatVec(const std::string& val);

template std::vector<Eigen::bfloat16>
Fp16BytesToFloatVec(const std::string& val);

template <typename T>
std::vector<T>
Bf16BytesToFloatVec(const std::string& val) {
  return Fp16BytesToFloatVecImpl<Eigen::bfloat16, T>(val);
}

template std::vector<float>
Bf16BytesToFloatVec(const std::string& val);

template std::vector<double>
Bf16BytesToFloatVec(const std::string& val);

template std::vector<Eigen::bfloat16>
Bf16BytesToFloatVec(const std::string& val);


template <typename Fp16T>
std::string
CreateBinaryStringFromFp16(const std::vector<Fp16T>& data) {
    static_assert(sizeof(Fp16T) == 2, "Fp16T should be 2 bytes");
    static_assert(std::is_same_v<Fp16T, Eigen::half> || std::is_same_v<Fp16T, Eigen::bfloat16>,
                  "Fp16T should be Eigen::half or Eigen::bfloat16");
    if constexpr (BYTE_ORDER == LITTLE_ENDIAN) {
        return std::string(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(Fp16T));
    } else {
        // Big endian
        std::string ret;
        ret.reserve(data.size() * sizeof(Fp16T));
        for (const Fp16T item : data) {
            union {
                uint16_t u16;
                char bytes[2];
            } value;
            if constexpr (std::is_same_v<Fp16T, Eigen::half>) {
                value.u16 = Eigen::half_impl::raw_half_as_uint16(item);
            } else {
                value.u16 = Eigen::bfloat16_impl::raw_bfloat16_as_uint16(item);
            }
            ret.push_back(value.bytes[1]);
            ret.push_back(value.bytes[0]);
        }
        return ret;
    }
}

template <typename FloatT>
std::string
CreateBinaryString(const std::vector<FloatT>& data) {
    static_assert(std::is_same_v<FloatT, float> || std::is_same_v<FloatT, double>, "FloatT should be float or double");
    std::string ret;
    ret.reserve(data.size() * 2);
    for (const auto item : data) {
        Fp16T fp16(static_cast<float>(item));
        union {
            uint16_t u16;
            char bytes[2];
        } value;
        if constexpr (std::is_same_v<Fp16T, Eigen::half>) {
            value.u16 = Eigen::half_impl::raw_half_as_uint16(fp16);
        } else {
            value.u16 = Eigen::bfloat16_impl::raw_bfloat16_as_uint16(fp16);
        }
        if constexpr (BYTE_ORDER == LITTLE_ENDIAN) {
            ret.push_back(value.bytes[0]);
            ret.push_back(value.bytes[1]);
        } else {
            ret.push_back(value.bytes[1]);
            ret.push_back(value.bytes[0]);
        }
    }
    return ret;
}

}  // namespace milvus
