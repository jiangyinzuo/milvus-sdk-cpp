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

#pragma once

#include <string>
#include <vector>

namespace Eigen {
struct half;
struct bfloat16;
}  // namespace Eigen

namespace milvus {

template <typename T>
std::vector<T>
Fp16BytesToFloatVec(const std::string& val);

extern template std::vector<float>
Fp16BytesToFloatVec(const std::string& val);

extern template std::vector<double>
Fp16BytesToFloatVec(const std::string& val);

extern template std::vector<Eigen::bfloat16>
Fp16BytesToFloatVec(const std::string& val);

template <typename T>
std::vector<T>
Bf16BytesToFloatVec(const std::string& val);

extern template std::vector<float>
Bf16BytesToFloatVec(const std::string& val);

extern template std::vector<double>
Bf16BytesToFloatVec(const std::string& val);

extern template std::vector<Eigen::bfloat16>
Bf16BytesToFloatVec(const std::string& val);

/*
 * Convert fp16 vector to bytes
 */
std::string
Fp16VecToBytes(const std::vector<Eigen::half>& data);

/*
 * Convert bf16 vector to bytes
 */
std::string
Bf16VecToBytes(const std::vector<Eigen::bfloat16>& data);

std::string
FloatToFp16Bytes(const std::vector<float>& data);

std::string
FloatToBf16Bytes(const std::vector<float>& data);

std::string
DoubleToFp16Bytes(const std::vector<double>& data);

std::string
DoubleToBf16Bytes(const std::vector<double>& data);

}  // namespace milvus
