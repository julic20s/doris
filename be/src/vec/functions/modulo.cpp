// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
// This file is copied from
// https://github.com/ClickHouse/ClickHouse/blob/master/src/Functions/Modulo.cpp
// and modified by Doris

#if USE_AVX2
#define LIBDIVIDE_AVX2
#endif
#include <libdivide.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <ranges>
#include <utility>

#include "runtime/decimalv2_value.h"
#include "vec/columns/column_vector.h"
#include "vec/core/types.h"
#include "vec/data_types/number_traits.h"
#include "vec/functions/function_binary_arithmetic.h"
#include "vec/functions/simple_function_factory.h"

namespace doris::vectorized {

template <typename A, typename B>
struct ModuloImpl {
    using ResultType = typename NumberTraits::ResultOfModulo<A, B>::Type;
    using Traits = NumberTraits::BinaryOperatorTraits<A, B>;

    template <typename Result = ResultType>
    static void apply(const typename Traits::ArrayA& a, B b,
                      typename ColumnVector<Result>::Container& c,
                      typename Traits::ArrayNull& null_map) {
        size_t size = c.size();
        UInt8 is_null = b == 0;
        memset(null_map.data(), is_null, sizeof(UInt8) * size);

        if (!is_null) {
            for (size_t i = 0; i < size; i++) {
                if constexpr (std::is_floating_point_v<ResultType>) {
                    c[i] = std::fmod((double)a[i], (double)b);
                } else {
                    c[i] = a[i] % b;
                }
            }
        }
    }

    template <typename Result = ResultType>
    static inline Result apply(A a, B b, UInt8& is_null) {
        is_null = b == 0;
        b += is_null;

        if constexpr (std::is_floating_point_v<Result>) {
            return std::fmod((double)a, (double)b);
        } else {
            return a % b;
        }
    }

    template <typename Result = DecimalV2Value>
    static inline DecimalV2Value apply(DecimalV2Value a, DecimalV2Value b, UInt8& is_null) {
        is_null = b == DecimalV2Value(0);
        return a % (b + DecimalV2Value(is_null));
    }
};

/* template <typename A, typename B>
struct ModuloImpl {
    using ResultType = typename NumberTraits::ResultOfModulo<A, B>::Type;
    using Traits = NumberTraits::BinaryOperatorTraits<A, B>;

    template <typename Result = ResultType>
    static void apply(const typename Traits::ArrayA& a, B b,
                      typename ColumnVector<Result>::Container& c,
                      typename Traits::ArrayNull& null_map) {
        size_t size = c.size();
        UInt8 is_null = b == 0;
        memset(null_map.data(), is_null, sizeof(UInt8) * size);
        if (is_null) {
            return;
        }

        if constexpr (std::is_integral_v<A> && std::is_integral_v<B>) {
            // Modulo of division by negative number is the same as the positive number.
            if (std::is_signed_v<B> && b < 0) {
                b = -b;
            }
            // Modulo with too small divisor.
            [[unlikely]] if (b == 1) {
                std::ranges::fill(c, 0);
                return;
            }

            // Modulo with too large divisor.
            [[unlikely]] if (b > std::numeric_limits<A>::max()) {
                for (size_t i = 0; i < size; ++i) {
                    c[i] = static_cast<Result>(a[i]);
                }
                return;
            }

            if ((b & (b - 1)) == 0) {
                // Modulo with pow2.
                auto mask = b - 1;
                for (size_t i = 0; i < size; ++i) {
                    c[i] = static_cast<Result>(a[i] & mask);
                }
                return;
            }

            using DivType = decltype(a[0] / b);
            // Used to fit Int8/UInt8.
            using Promoted = std::conditional_t<(sizeof(DivType) > 1), DivType, Int16>;
            constexpr bool use_lib = requires {
                libdivide::dispatcher<Promoted, libdivide::BRANCHFULL>(static_cast<Promoted>(b))
                        .divide(static_cast<Promoted>(a[0]));
            };
            if constexpr (use_lib) {
                libdivide::divider<Promoted> d(static_cast<Promoted>(b));
                for (size_t i = 0; i < size; ++i) {
                    c[i] = static_cast<Result>(static_cast<Promoted>(a[i]) -
                                               (static_cast<Promoted>(a[i]) / d) * b);
                }
                return;
            }
            for (size_t i = 0; i < size; ++i) {
                c[i] = static_cast<Result>(a[i] % b);
            }
        } else {
            static_assert(std::is_floating_point_v<Result>,
                          "If one of the inputs is a floating-point number, result must be "
                          "floating-point number.");
            auto fb = static_cast<Result>(b);
            for (size_t i = 0; i < size; ++i) {
                c[i] = fast_fmod(static_cast<Result>(a[i]), fb);
            }
        }
    }

    template <typename Result = ResultType>
    static Result apply(A a, B b, UInt8& is_null) {
        is_null = b == 0;
        b += is_null;

        if constexpr (std::is_floating_point_v<Result>) {
            return fast_fmod((double)a, (double)b);
        } else {
            return a % b;
        }
    }

    template <typename Result = DecimalV2Value>
    static inline DecimalV2Value apply(DecimalV2Value a, DecimalV2Value b, UInt8& is_null) {
        is_null = b == DecimalV2Value(0);
        return a % (b + DecimalV2Value(is_null));
    }

    template <std::floating_point T>
    static T fast_fmod(T a, T b) {
        /// This computation is similar to `fmod` but the latter is not inlined and has 40 times worse performance.
        if constexpr (std::is_same_v<T, float>) {
            return a - std::truncf(a / b) * b;
        }
        return a - std::trunc(a / b) * b;
    }
}; */

template <typename A, typename B>
struct PModuloImpl {
    using ResultType = typename NumberTraits::ResultOfModulo<A, B>::Type;
    using Traits = NumberTraits::BinaryOperatorTraits<A, B>;

    template <typename Result = ResultType>
    static void apply(const typename Traits::ArrayA& a, B b,
                      typename ColumnVector<Result>::Container& c,
                      typename Traits::ArrayNull& null_map) {
        size_t size = c.size();
        UInt8 is_null = b == 0;
        memset(null_map.data(), is_null, size);

        if (!is_null) {
            for (size_t i = 0; i < size; i++) {
                if constexpr (std::is_floating_point_v<ResultType>) {
                    c[i] = std::fmod(std::fmod((double)a[i], (double)b) + (double)b, double(b));
                } else {
                    c[i] = (a[i] % b + b) % b;
                }
            }
        }
    }

    template <typename Result = ResultType>
    static inline Result apply(A a, B b, UInt8& is_null) {
        is_null = b == 0;
        b += is_null;

        if constexpr (std::is_floating_point_v<Result>) {
            return std::fmod(std::fmod((double)a, (double)b) + (double)b, (double)b);
        } else {
            return (a % b + b) % b;
        }
    }

    template <typename Result = DecimalV2Value>
    static inline DecimalV2Value apply(DecimalV2Value a, DecimalV2Value b, UInt8& is_null) {
        is_null = b == DecimalV2Value(0);
        b += DecimalV2Value(is_null);
        return (a % b + b) % b;
    }
};

struct NameModulo {
    static constexpr auto name = "mod";
};
struct NamePModulo {
    static constexpr auto name = "pmod";
};

using FunctionModulo = FunctionBinaryArithmetic<ModuloImpl, NameModulo, true>;
using FunctionPModulo = FunctionBinaryArithmetic<PModuloImpl, NamePModulo, true>;

void register_function_modulo(SimpleFunctionFactory& factory) {
    factory.register_function<FunctionModulo>();
    factory.register_function<FunctionPModulo>();
    factory.register_alias("mod", "fmod");
}

} // namespace doris::vectorized
