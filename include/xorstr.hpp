/*
 * Copyright 2017 - 2021 Justas Masiulis
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

#ifndef JM_XORSTR_HPP
#define JM_XORSTR_HPP

#if defined(_M_ARM64) || defined(__aarch64__) || defined(_M_ARM) || defined(__arm__)
#include <arm_neon.h>
#elif defined(_M_X64) || defined(__amd64__) || defined(_M_IX86) || defined(__i386__)
#include <immintrin.h>
#else
#error Unsupported platform
#endif

#include <cstdint>
#include <cstddef>
#include <utility>
#include <type_traits>

// ---------------------------------------------------------------------------
// AVX2 auto-detection
// ---------------------------------------------------------------------------
// The original code gated its 256-bit path on JM_XORSTR_DISABLE_AVX_INTRINSICS
// and used it by default. That path calls _mm256_xor_si256, which is an AVX2
// intrinsic (AVX1 only added 256-bit float/double ops, not integer ones), so
// building this header with default flags on an AVX-only CPU (e.g. Sandy
// Bridge / Ivy Bridge, like an i5-2400) would compile fine but crash at
// runtime with SIGILL, unless the user manually opted out.
//
// Since this is a compile-time/inlined template library, "does the target
// have AVX2" is decided at compile time by the compiler flags you build
// with (-mavx2 / -march=native / -march=haswell-or-later on GCC/Clang,
// /arch:AVX2 on MSVC). All of these define the __AVX2__ macro for you, so
// we detect that instead of assuming AVX2 is present.
//
//   * Build normally (no -mavx2)      -> falls back to SSE, safe on any x86-64 CPU.
//   * Build with -mavx2 / -march=native on an AVX2-capable machine
//                                      -> automatically takes the faster 256-bit path.
//   * JM_XORSTR_DISABLE_AVX_INTRINSICS -> force SSE regardless of __AVX2__.
//   * JM_XORSTR_FORCE_AVX2             -> force the AVX2 path regardless of
//                                         __AVX2__ (only for testing on a
//                                         known-AVX2 machine; will SIGILL if
//                                         the CPU actually lacks AVX2).
#if defined(JM_XORSTR_DISABLE_AVX_INTRINSICS)
#   define JM_XORSTR_USE_AVX2 0
#elif defined(JM_XORSTR_FORCE_AVX2)
#   define JM_XORSTR_USE_AVX2 1
#elif defined(__AVX2__)
#   define JM_XORSTR_USE_AVX2 1
#else
#   define JM_XORSTR_USE_AVX2 0
#endif

// ---------------------------------------------------------------------------
// True runtime CPU dispatch (GCC/Clang, x86/x86-64 only)
// ---------------------------------------------------------------------------
// Everything above decides AVX2 usage at *compile* time, from the flags you
// built with. That means a binary built without -mavx2 never uses AVX2 even
// on a machine that has it, and a binary built with -mavx2 will crash with
// SIGILL if it's ever run on a machine without AVX2 (e.g. an i5-2400).
//
// If you want a single binary that runs correctly on both, and automatically
// takes the faster AVX2 path only on CPUs that actually support it, that
// needs a *runtime* check (CPUID) plus two compiled versions of the XOR
// routine. GCC and Clang support this via per-function `target("avx2")`
// attributes (function multiversioning) — the AVX2 version can be compiled
// into the binary without needing a blanket -mavx2 for the whole file, and
// you choose which one runs via __builtin_cpu_supports at runtime.
//
// This is only available for GCC/Clang on x86/x86-64; MSVC has no equivalent
// per-function target attribute (its AVX2 intrinsics require /arch:AVX2 for
// the whole translation unit), so on MSVC this falls back to the compile-time
// JM_XORSTR_USE_AVX2 decision above. ARM doesn't need this at all: NEON is a
// mandatory baseline feature on AArch64, so there's nothing to detect.
#if (defined(_M_X64) || defined(__amd64__) || defined(_M_IX86) || defined(__i386__)) \
    && (defined(__GNUC__) || defined(__clang__)) \
    && !defined(JM_XORSTR_DISABLE_AVX_INTRINSICS) \
    && !defined(JM_XORSTR_FORCE_AVX2) \
    && !defined(JM_XORSTR_DISABLE_RUNTIME_DISPATCH)
#   define JM_XORSTR_RUNTIME_DISPATCH 1
#else
#   define JM_XORSTR_RUNTIME_DISPATCH 0
#endif

#define xorstr(str) ::jm::xor_string([]() { return str; }, std::integral_constant<std::size_t, sizeof(str) / sizeof(*str)>{}, std::make_index_sequence<::jm::detail::_buffer_size<sizeof(str)>()>{})
#define xorstr_(str) xorstr(str).crypt_get()

#ifdef _MSC_VER
#define XORSTR_FORCEINLINE __forceinline
#else
#define XORSTR_FORCEINLINE __attribute__((always_inline)) inline
#endif

namespace jm {

    namespace detail {

        template<std::size_t Size>
        XORSTR_FORCEINLINE constexpr std::size_t _buffer_size()
        {
            return ((Size / 16) + (Size % 16 != 0)) * 2;
        }

        template<std::uint32_t Seed>
        XORSTR_FORCEINLINE constexpr std::uint32_t key4() noexcept
        {
            std::uint32_t value = Seed;
            for(char c : __TIME__)
                value = static_cast<std::uint32_t>((value ^ c) * 16777619ull);
            return value;
        }

        template<std::size_t S>
        XORSTR_FORCEINLINE constexpr std::uint64_t key8()
        {
            constexpr auto first_part  = key4<2166136261 + S>();
            constexpr auto second_part = key4<first_part>();
            return (static_cast<std::uint64_t>(first_part) << 32) | second_part;
        }

        // loads up to 8 characters of string into uint64 and xors it with the key
        template<std::size_t N, class CharT>
        XORSTR_FORCEINLINE constexpr std::uint64_t
        load_xored_str8(std::uint64_t key, std::size_t idx, const CharT* str) noexcept
        {
            using cast_type = typename std::make_unsigned<CharT>::type;
            constexpr auto value_size = sizeof(CharT);
            constexpr auto idx_offset = 8 / value_size;

            std::uint64_t value = key;
            for(std::size_t i = 0; i < idx_offset && i + idx * idx_offset < N; ++i)
                value ^=
                    (std::uint64_t{ static_cast<cast_type>(str[i + idx * idx_offset]) }
                     << ((i % idx_offset) * 8 * value_size));

            return value;
        }

        // forces compiler to use registers instead of stuffing constants in rdata
        XORSTR_FORCEINLINE std::uint64_t load_from_reg(std::uint64_t value) noexcept
        {
#if defined(__clang__) || defined(__GNUC__)
            asm("" : "=r"(value) : "0"(value) :);
            return value;
#else
            volatile std::uint64_t reg = value;
            return reg;
#endif
        }

#if JM_XORSTR_RUNTIME_DISPATCH

        // Checked once (function-local static init is thread-safe in C++11+)
        // and cheap on every call after that - just a load of a cached bool.
        inline bool cpu_has_avx2() noexcept
        {
            static const bool result = []() noexcept {
                __builtin_cpu_init();
                return __builtin_cpu_supports("avx2") != 0;
            }();
            return result;
        }

        // Compiled with AVX2 codegen enabled for just this function, even
        // though the rest of the translation unit is not built with -mavx2.
        // Handles any even element count (the storage/keys arrays this
        // library generates are always sized in multiples of 2 uint64_t's).
        __attribute__((target("avx2")))
        inline void xor_buffer_avx2(std::uint64_t*       storage,
                                     const std::uint64_t* keys,
                                     std::size_t          count64) noexcept
        {
            std::size_t i = 0;
            for(; i + 4 <= count64; i += 4) {
                __m256i s = _mm256_load_si256(reinterpret_cast<const __m256i*>(storage + i));
                __m256i k = _mm256_load_si256(reinterpret_cast<const __m256i*>(keys + i));
                _mm256_store_si256(reinterpret_cast<__m256i*>(storage + i), _mm256_xor_si256(s, k));
            }
            if(i + 2 <= count64) {
                __m128i s = _mm_load_si128(reinterpret_cast<const __m128i*>(storage + i));
                __m128i k = _mm_load_si128(reinterpret_cast<const __m128i*>(keys + i));
                _mm_store_si128(reinterpret_cast<__m128i*>(storage + i), _mm_xor_si128(s, k));
            }
        }

        // Baseline path: plain SSE2, guaranteed available on every x86-64 CPU
        // (including AVX-only ones like an i5-2400), no special target needed.
        inline void xor_buffer_sse(std::uint64_t*       storage,
                                    const std::uint64_t* keys,
                                    std::size_t          count64) noexcept
        {
            std::size_t i = 0;
            for(; i + 2 <= count64; i += 2) {
                __m128i s = _mm_load_si128(reinterpret_cast<const __m128i*>(storage + i));
                __m128i k = _mm_load_si128(reinterpret_cast<const __m128i*>(keys + i));
                _mm_store_si128(reinterpret_cast<__m128i*>(storage + i), _mm_xor_si128(s, k));
            }
        }

        XORSTR_FORCEINLINE void xor_buffer_dispatch(std::uint64_t*       storage,
                                                      const std::uint64_t* keys,
                                                      std::size_t          count64) noexcept
        {
            if(cpu_has_avx2())
                xor_buffer_avx2(storage, keys, count64);
            else
                xor_buffer_sse(storage, keys, count64);
        }

#endif // JM_XORSTR_RUNTIME_DISPATCH

    } // namespace detail

    template<class CharT, std::size_t Size, class Keys, class Indices>
    class xor_string;

    template<class CharT, std::size_t Size, std::uint64_t... Keys, std::size_t... Indices>
    class xor_string<CharT, Size, std::integer_sequence<std::uint64_t, Keys...>, std::index_sequence<Indices...>> {
#if JM_XORSTR_RUNTIME_DISPATCH || JM_XORSTR_USE_AVX2
        constexpr static inline std::uint64_t alignment = ((Size > 16) ? 32 : 16);    
#else
        constexpr static inline std::uint64_t alignment = 16;
#endif

        alignas(alignment) std::uint64_t _storage[sizeof...(Keys)];

    public:
        using value_type    = CharT;
        using size_type     = std::size_t;
        using pointer       = CharT*;
        using const_pointer = const CharT*;

        template<class L>
        XORSTR_FORCEINLINE xor_string(L l, std::integral_constant<std::size_t, Size>, std::index_sequence<Indices...>) noexcept
            : _storage{ ::jm::detail::load_from_reg((std::integral_constant<std::uint64_t, detail::load_xored_str8<Size>(Keys, Indices, l())>::value))... }
        {}

        XORSTR_FORCEINLINE constexpr size_type size() const noexcept
        {
            return Size - 1;
        }

        XORSTR_FORCEINLINE void crypt() noexcept
        {
            // everything is inlined by hand because a certain compiler with a certain linker is _very_ slow
#if defined(__clang__)
            alignas(alignment)
                std::uint64_t arr[]{ ::jm::detail::load_from_reg(Keys)... };
            std::uint64_t*    keys =
                (std::uint64_t*)::jm::detail::load_from_reg((std::uint64_t)arr);
#else
            alignas(alignment) std::uint64_t keys[]{ ::jm::detail::load_from_reg(Keys)... };
#endif

#if defined(_M_ARM64) || defined(__aarch64__) || defined(_M_ARM) || defined(__arm__)
#if defined(__clang__)
            ((Indices >= sizeof(_storage) / 16 ? static_cast<void>(0) : __builtin_neon_vst1q_v(
                                    reinterpret_cast<uint64_t*>(_storage) + Indices * 2,
                                    veorq_u64(__builtin_neon_vld1q_v(reinterpret_cast<const uint64_t*>(_storage) + Indices * 2, 51),
                                              __builtin_neon_vld1q_v(reinterpret_cast<const uint64_t*>(keys) + Indices * 2, 51)),
                                    51)), ...);
#else // GCC, MSVC
            ((Indices >= sizeof(_storage) / 16 ? static_cast<void>(0) : vst1q_u64(
                        reinterpret_cast<uint64_t*>(_storage) + Indices * 2,
                        veorq_u64(vld1q_u64(reinterpret_cast<const uint64_t*>(_storage) + Indices * 2),
                                  vld1q_u64(reinterpret_cast<const uint64_t*>(keys) + Indices * 2)))), ...);
#endif
#elif JM_XORSTR_RUNTIME_DISPATCH
            ::jm::detail::xor_buffer_dispatch(_storage, keys, sizeof...(Keys));
#elif JM_XORSTR_USE_AVX2
            ((Indices >= sizeof(_storage) / 32 ? static_cast<void>(0) : _mm256_store_si256(
                reinterpret_cast<__m256i*>(_storage) + Indices,
                _mm256_xor_si256(
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(_storage) + Indices),
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(keys) + Indices)))), ...);

            if constexpr(sizeof(_storage) % 32 != 0)
                _mm_store_si128(
                    reinterpret_cast<__m128i*>(_storage + sizeof...(Keys) - 2),
                    _mm_xor_si128(_mm_load_si128(reinterpret_cast<const __m128i*>(_storage + sizeof...(Keys) - 2)),
                                  _mm_load_si128(reinterpret_cast<const __m128i*>(keys + sizeof...(Keys) - 2))));
#else
        ((Indices >= sizeof(_storage) / 16 ? static_cast<void>(0) : _mm_store_si128(
            reinterpret_cast<__m128i*>(_storage) + Indices,
            _mm_xor_si128(_mm_load_si128(reinterpret_cast<const __m128i*>(_storage) + Indices),
                          _mm_load_si128(reinterpret_cast<const __m128i*>(keys) + Indices)))), ...);
#endif
        }

        XORSTR_FORCEINLINE const_pointer get() const noexcept
        {
            return reinterpret_cast<const_pointer>(_storage);
        }

        XORSTR_FORCEINLINE pointer get() noexcept
        {
            return reinterpret_cast<pointer>(_storage);
        }

        XORSTR_FORCEINLINE pointer crypt_get() noexcept
        {
            // crypt() is inlined by hand because a certain compiler with a certain linker is _very_ slow
#if defined(__clang__)
            alignas(alignment)
                std::uint64_t arr[]{ ::jm::detail::load_from_reg(Keys)... };
            std::uint64_t*    keys =
                (std::uint64_t*)::jm::detail::load_from_reg((std::uint64_t)arr);
#else
            alignas(alignment) std::uint64_t keys[]{ ::jm::detail::load_from_reg(Keys)... };
#endif

#if defined(_M_ARM64) || defined(__aarch64__) || defined(_M_ARM) || defined(__arm__)
#if defined(__clang__)
            ((Indices >= sizeof(_storage) / 16 ? static_cast<void>(0) : __builtin_neon_vst1q_v(
                                    reinterpret_cast<uint64_t*>(_storage) + Indices * 2,
                                    veorq_u64(__builtin_neon_vld1q_v(reinterpret_cast<const uint64_t*>(_storage) + Indices * 2, 51),
                                              __builtin_neon_vld1q_v(reinterpret_cast<const uint64_t*>(keys) + Indices * 2, 51)),
                                    51)), ...);
#else // GCC, MSVC
            ((Indices >= sizeof(_storage) / 16 ? static_cast<void>(0) : vst1q_u64(
                        reinterpret_cast<uint64_t*>(_storage) + Indices * 2,
                        veorq_u64(vld1q_u64(reinterpret_cast<const uint64_t*>(_storage) + Indices * 2),
                                  vld1q_u64(reinterpret_cast<const uint64_t*>(keys) + Indices * 2)))), ...);
#endif
#elif JM_XORSTR_RUNTIME_DISPATCH
            ::jm::detail::xor_buffer_dispatch(_storage, keys, sizeof...(Keys));
#elif JM_XORSTR_USE_AVX2
            ((Indices >= sizeof(_storage) / 32 ? static_cast<void>(0) : _mm256_store_si256(
                reinterpret_cast<__m256i*>(_storage) + Indices,
                _mm256_xor_si256(
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(_storage) + Indices),
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(keys) + Indices)))), ...);

            if constexpr(sizeof(_storage) % 32 != 0)
                _mm_store_si128(
                    reinterpret_cast<__m128i*>(_storage + sizeof...(Keys) - 2),
                    _mm_xor_si128(_mm_load_si128(reinterpret_cast<const __m128i*>(_storage + sizeof...(Keys) - 2)),
                                  _mm_load_si128(reinterpret_cast<const __m128i*>(keys + sizeof...(Keys) - 2))));
#else
        ((Indices >= sizeof(_storage) / 16 ? static_cast<void>(0) : _mm_store_si128(
            reinterpret_cast<__m128i*>(_storage) + Indices,
            _mm_xor_si128(_mm_load_si128(reinterpret_cast<const __m128i*>(_storage) + Indices),
                          _mm_load_si128(reinterpret_cast<const __m128i*>(keys) + Indices)))), ...);
#endif

            return (pointer)(_storage);
        }
    };

    template<class L, std::size_t Size, std::size_t... Indices>
    xor_string(L l, std::integral_constant<std::size_t, Size>, std::index_sequence<Indices...>) -> xor_string<
                std::remove_const_t<std::remove_reference_t<decltype(l()[0])>>,
                Size,
                std::integer_sequence<std::uint64_t, detail::key8<Indices>()...>,
                std::index_sequence<Indices...>>;

} // namespace jm

#endif // include guard
