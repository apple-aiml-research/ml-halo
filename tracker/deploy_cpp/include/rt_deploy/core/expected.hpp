//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// Minimal Result<T, E> for off-hot-path error returns (config / init paths).
//
// A small stand-in for C++23 std::expected, which the C++20 toolchains this
// project targets (e.g. GCC 11 on JetPack 6) do not provide. Errors are built
// either with `Result<T>::Error{msg}` or with `core::unexpected(msg)`.
// Never used on the hot path.

#include <string>
#include <utility>
#include <variant>

namespace rt_deploy::core {

// Non-nested wrapper so the same `core::unexpected(...)` call is convertible
// into any `Result<T, E>` regardless of T. Mirrors how std::unexpected<E>
// converts into any std::expected<T, E>.
template <typename E>
struct Unexpected {
    E value;
};

template <typename T, typename E = std::string>
class Result {
public:
    // NOLINTNEXTLINE(google-explicit-constructor)
    Result(T v) : data_{std::in_place_index<0>, std::move(v)} {}

    struct Error {
        E value;
    };
    // NOLINTNEXTLINE(google-explicit-constructor)
    Result(Error e) : data_{std::in_place_index<1>, std::move(e.value)} {}
    // NOLINTNEXTLINE(google-explicit-constructor)
    Result(Unexpected<E> u) : data_{std::in_place_index<1>, std::move(u.value)} {}

    [[nodiscard]] bool has_value() const noexcept { return data_.index() == 0; }
    [[nodiscard]] explicit operator bool() const noexcept { return has_value(); }

    [[nodiscard]] T& value() & { return std::get<0>(data_); }
    [[nodiscard]] const T& value() const& { return std::get<0>(data_); }
    [[nodiscard]] T&& value() && { return std::get<0>(std::move(data_)); }

    [[nodiscard]] const E& error() const& { return std::get<1>(data_); }
    [[nodiscard]] E&&      error() && { return std::get<1>(std::move(data_)); }

    T& operator*() & { return value(); }
    const T& operator*() const& { return value(); }
    T* operator->() { return &value(); }
    const T* operator->() const { return &value(); }

private:
    std::variant<T, E> data_;
};

template <typename E>
[[nodiscard]] Unexpected<E> unexpected(E e) {
    return Unexpected<E>{std::move(e)};
}

// Specialization for void-value Result.
template <typename E>
class Result<void, E> {
public:
    Result() = default;
    struct Error {
        E value;
    };
    // NOLINTNEXTLINE(google-explicit-constructor)
    Result(Error e) : error_{std::move(e.value)}, ok_{false} {}
    // NOLINTNEXTLINE(google-explicit-constructor)
    Result(Unexpected<E> u) : error_{std::move(u.value)}, ok_{false} {}

    [[nodiscard]] bool has_value() const noexcept { return ok_; }
    [[nodiscard]] explicit operator bool() const noexcept { return ok_; }
    [[nodiscard]] const E& error() const& { return error_; }

private:
    E    error_{};
    bool ok_{true};
};

}  // namespace rt_deploy::core
