#ifndef SHOGUN_TYPE_LIST_H
#define SHOGUN_TYPE_LIST_H

// Copyright 2008 Google Inc.
// All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// modified by Gil Hoben
// This is a stripped down version of a generated "gtest-type-util.h"

// A unique type used as the default value for the arguments of class
// template Types.  This allows us to simulate variadic templates
// (e.g. Types<int>, Type<int, double>, and etc), which C++ doesn't
// support directly.
struct None {};

// The following family of struct and struct templates are used to
// represent type lists.  In particular, TypesN<T1, T2, ..., TN>
// represents a type list with N types (T1, T2, ..., and TN) in it.
// Except for Types0, every struct in the family has two member types:
// Head for the first type in the list, and Tail for the rest of the
// list.

// The empty type list.
struct Types0 {
    typedef Types0 Tail;
};

// Type lists of length 1, 2, 3, and so on.

template <typename T1> struct Types1 {
    typedef T1 Head;
    typedef Types0 Tail;
};
template <typename T1, typename T2> struct Types2 {
    typedef T1 Head;
    typedef Types1<T2> Tail;
};

template <typename T1, typename T2, typename T3> struct Types3 {
    typedef T1 Head;
    typedef Types2<T2, T3> Tail;
};

template <typename T1, typename T2, typename T3, typename T4> struct Types4 {
    typedef T1 Head;
    typedef Types3<T2, T3, T4> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5>
struct Types5 {
    typedef T1 Head;
    typedef Types4<T2, T3, T4, T5> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6>
struct Types6 {
    typedef T1 Head;
    typedef Types5<T2, T3, T4, T5, T6> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7>
struct Types7 {
    typedef T1 Head;
    typedef Types6<T2, T3, T4, T5, T6, T7> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8>
struct Types8 {
    typedef T1 Head;
    typedef Types7<T2, T3, T4, T5, T6, T7, T8> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9>
struct Types9 {
    typedef T1 Head;
    typedef Types8<T2, T3, T4, T5, T6, T7, T8, T9> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10>
struct Types10 {
    typedef T1 Head;
    typedef Types9<T2, T3, T4, T5, T6, T7, T8, T9, T10> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11>
struct Types11 {
    typedef T1 Head;
    typedef Types10<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12>
struct Types12 {
    typedef T1 Head;
    typedef Types11<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13>
struct Types13 {
    typedef T1 Head;
    typedef Types12<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14>
struct Types14 {
    typedef T1 Head;
    typedef Types13<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15>
struct Types15 {
    typedef T1 Head;
    typedef Types14<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16>
struct Types16 {
    typedef T1 Head;
    typedef Types15<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17>
struct Types17 {
    typedef T1 Head;
    typedef Types16<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18>
struct Types18 {
    typedef T1 Head;
    typedef Types17<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19>
struct Types19 {
    typedef T1 Head;
    typedef Types18<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20>
struct Types20 {
    typedef T1 Head;
    typedef Types19<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21>
struct Types21 {
    typedef T1 Head;
    typedef Types20<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22>
struct Types22 {
    typedef T1 Head;
    typedef Types21<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23>
struct Types23 {
    typedef T1 Head;
    typedef Types22<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24>
struct Types24 {
    typedef T1 Head;
    typedef Types23<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25>
struct Types25 {
    typedef T1 Head;
    typedef Types24<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26>
struct Types26 {
    typedef T1 Head;
    typedef Types25<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27>
struct Types27 {
    typedef T1 Head;
    typedef Types26<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28>
struct Types28 {
    typedef T1 Head;
    typedef Types27<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
            T28>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29>
struct Types29 {
    typedef T1 Head;
    typedef Types28<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
            T28, T29>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30>
struct Types30 {
    typedef T1 Head;
    typedef Types29<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
            T28, T29, T30>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31>
struct Types31 {
    typedef T1 Head;
    typedef Types30<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
            T28, T29, T30, T31>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32>
struct Types32 {
    typedef T1 Head;
    typedef Types31<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
            T28, T29, T30, T31, T32>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33>
struct Types33 {
    typedef T1 Head;
    typedef Types32<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
            T28, T29, T30, T31, T32, T33>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34>
struct Types34 {
    typedef T1 Head;
    typedef Types33<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
            T28, T29, T30, T31, T32, T33, T34>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35>
struct Types35 {
    typedef T1 Head;
    typedef Types34<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
            T28, T29, T30, T31, T32, T33, T34, T35>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36>
struct Types36 {
    typedef T1 Head;
    typedef Types35<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
            T28, T29, T30, T31, T32, T33, T34, T35, T36>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37>
struct Types37 {
    typedef T1 Head;
    typedef Types36<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
            T28, T29, T30, T31, T32, T33, T34, T35, T36, T37>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38>
struct Types38 {
    typedef T1 Head;
    typedef Types37<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
            T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38, typename T39>
struct Types39 {
    typedef T1 Head;
    typedef Types38<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
            T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38, typename T39, typename T40>
struct Types40 {
    typedef T1 Head;
    typedef Types39<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
            T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39,
            T40>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38, typename T39, typename T40,
        typename T41>
struct Types41 {
    typedef T1 Head;
    typedef Types40<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
            T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39,
            T40, T41>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38, typename T39, typename T40,
        typename T41, typename T42>
struct Types42 {
    typedef T1 Head;
    typedef Types41<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
            T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39,
            T40, T41, T42>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38, typename T39, typename T40,
        typename T41, typename T42, typename T43>
struct Types43 {
    typedef T1 Head;
    typedef Types42<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
            T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39,
            T40, T41, T42, T43>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38, typename T39, typename T40,
        typename T41, typename T42, typename T43, typename T44>
struct Types44 {
    typedef T1 Head;
    typedef Types43<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
            T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39,
            T40, T41, T42, T43, T44>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38, typename T39, typename T40,
        typename T41, typename T42, typename T43, typename T44, typename T45>
struct Types45 {
    typedef T1 Head;
    typedef Types44<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
            T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39,
            T40, T41, T42, T43, T44, T45>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38, typename T39, typename T40,
        typename T41, typename T42, typename T43, typename T44, typename T45,
        typename T46>
struct Types46 {
    typedef T1 Head;
    typedef Types45<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
            T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39,
            T40, T41, T42, T43, T44, T45, T46>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38, typename T39, typename T40,
        typename T41, typename T42, typename T43, typename T44, typename T45,
        typename T46, typename T47>
struct Types47 {
    typedef T1 Head;
    typedef Types46<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
            T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39,
            T40, T41, T42, T43, T44, T45, T46, T47>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38, typename T39, typename T40,
        typename T41, typename T42, typename T43, typename T44, typename T45,
        typename T46, typename T47, typename T48>
struct Types48 {
    typedef T1 Head;
    typedef Types47<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
            T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39,
            T40, T41, T42, T43, T44, T45, T46, T47, T48>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38, typename T39, typename T40,
        typename T41, typename T42, typename T43, typename T44, typename T45,
        typename T46, typename T47, typename T48, typename T49>
struct Types49 {
    typedef T1 Head;
    typedef Types48<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
            T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39,
            T40, T41, T42, T43, T44, T45, T46, T47, T48, T49>
            Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38, typename T39, typename T40,
        typename T41, typename T42, typename T43, typename T44, typename T45,
        typename T46, typename T47, typename T48, typename T49, typename T50>
struct Types50 {
    typedef T1 Head;
    typedef Types49<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
            T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
            T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39,
            T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50>
            Tail;
};

// We don't want to require the users to write TypesN<...> directly,
// as that would require them to count the length.  Types<...> is much
// easier to write, but generates horrible messages when there is a
// compiler error, as gcc insists on printing out each template
// argument, even if it has the default value (this means Types<int>
// will appear as Types<int, None, None, ..., None> in the compiler
// errors).
//
// Our solution is to combine the best part of the two approaches: a
// user would write Types<T1, ..., TN>, and Google Test will translate
// that to TypesN<T1, ..., TN> internally to make error messages
// readable.  The translation is done by the 'type' member of the
// Types template.
template <typename T1 = None, typename T2 = None, typename T3 = None,
        typename T4 = None, typename T5 = None, typename T6 = None,
        typename T7 = None, typename T8 = None, typename T9 = None,
        typename T10 = None, typename T11 = None, typename T12 = None,
        typename T13 = None, typename T14 = None, typename T15 = None,
        typename T16 = None, typename T17 = None, typename T18 = None,
        typename T19 = None, typename T20 = None, typename T21 = None,
        typename T22 = None, typename T23 = None, typename T24 = None,
        typename T25 = None, typename T26 = None, typename T27 = None,
        typename T28 = None, typename T29 = None, typename T30 = None,
        typename T31 = None, typename T32 = None, typename T33 = None,
        typename T34 = None, typename T35 = None, typename T36 = None,
        typename T37 = None, typename T38 = None, typename T39 = None,
        typename T40 = None, typename T41 = None, typename T42 = None,
        typename T43 = None, typename T44 = None, typename T45 = None,
        typename T46 = None, typename T47 = None, typename T48 = None,
        typename T49 = None, typename T50 = None>
struct Types {
    typedef Types50<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
            T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38,
            T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50>
            type;
};

template <>
struct Types<None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None> {
    typedef Types0 type;
};
template <typename T1>
struct Types<T1, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None> {
    typedef Types1<T1> type;
};
template <typename T1, typename T2>
struct Types<T1, T2, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None> {
    typedef Types2<T1, T2> type;
};
template <typename T1, typename T2, typename T3>
struct Types<T1, T2, T3, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None> {
    typedef Types3<T1, T2, T3> type;
};
template <typename T1, typename T2, typename T3, typename T4>
struct Types<T1, T2, T3, T4, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None> {
    typedef Types4<T1, T2, T3, T4> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5>
struct Types<T1, T2, T3, T4, T5, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None> {
    typedef Types5<T1, T2, T3, T4, T5> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6>
struct Types<T1, T2, T3, T4, T5, T6, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None> {
    typedef Types6<T1, T2, T3, T4, T5, T6> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7>
struct Types<T1, T2, T3, T4, T5, T6, T7, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None> {
    typedef Types7<T1, T2, T3, T4, T5, T6, T7> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None> {
    typedef Types8<T1, T2, T3, T4, T5, T6, T7, T8> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None> {
    typedef Types9<T1, T2, T3, T4, T5, T6, T7, T8, T9> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None> {
    typedef Types10<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None> {
    typedef Types11<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None> {
    typedef Types12<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None> {
    typedef Types13<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None> {
    typedef Types14<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None> {
    typedef Types15<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None> {
    typedef Types16<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None> {
    typedef Types17<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None> {
    typedef Types18<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None> {
    typedef Types19<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None> {
    typedef Types20<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None> {
    typedef Types21<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None> {
    typedef Types22<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None> {
    typedef Types23<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None> {
    typedef Types24<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None> {
    typedef Types25<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None> {
    typedef Types26<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None> {
    typedef Types27<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
            T27>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None> {
    typedef Types28<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
            T27, T28>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
        T29, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None> {
    typedef Types29<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
            T27, T28, T29>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
        T29, T30, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None> {
    typedef Types30<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
            T27, T28, T29, T30>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
        T29, T30, T31, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None> {
    typedef Types31<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
            T27, T28, T29, T30, T31>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
        T29, T30, T31, T32, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None> {
    typedef Types32<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
            T27, T28, T29, T30, T31, T32>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
        T29, T30, T31, T32, T33, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None> {
    typedef Types33<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
            T27, T28, T29, T30, T31, T32, T33>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
        T29, T30, T31, T32, T33, T34, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None> {
    typedef Types34<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
            T27, T28, T29, T30, T31, T32, T33, T34>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
        T29, T30, T31, T32, T33, T34, T35, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None> {
    typedef Types35<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
            T27, T28, T29, T30, T31, T32, T33, T34, T35>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
        T29, T30, T31, T32, T33, T34, T35, T36, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None> {
    typedef Types36<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
            T27, T28, T29, T30, T31, T32, T33, T34, T35, T36>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
        T29, T30, T31, T32, T33, T34, T35, T36, T37, None, None, None,
        None, None, None, None, None, None, None, None, None, None> {
    typedef Types37<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
            T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
        T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, None, None, None,
        None, None, None, None, None, None, None, None, None> {
    typedef Types38<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
            T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38, typename T39>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
        T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, None, None,
        None, None, None, None, None, None, None, None, None> {
    typedef Types39<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
            T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38,
            T39>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38, typename T39, typename T40>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
        T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, None,
        None, None, None, None, None, None, None, None, None> {
    typedef Types40<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
            T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38,
            T39, T40>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38, typename T39, typename T40,
        typename T41>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
        T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41,
        None, None, None, None, None, None, None, None, None> {
    typedef Types41<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
            T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38,
            T39, T40, T41>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38, typename T39, typename T40,
        typename T41, typename T42>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
        T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41,
        T42, None, None, None, None, None, None, None, None> {
    typedef Types42<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
            T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38,
            T39, T40, T41, T42>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38, typename T39, typename T40,
        typename T41, typename T42, typename T43>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
        T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41,
        T42, T43, None, None, None, None, None, None, None> {
    typedef Types43<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
            T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38,
            T39, T40, T41, T42, T43>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38, typename T39, typename T40,
        typename T41, typename T42, typename T43, typename T44>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
        T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41,
        T42, T43, T44, None, None, None, None, None, None> {
    typedef Types44<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
            T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38,
            T39, T40, T41, T42, T43, T44>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38, typename T39, typename T40,
        typename T41, typename T42, typename T43, typename T44, typename T45>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
        T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41,
        T42, T43, T44, T45, None, None, None, None, None> {
    typedef Types45<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
            T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38,
            T39, T40, T41, T42, T43, T44, T45>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38, typename T39, typename T40,
        typename T41, typename T42, typename T43, typename T44, typename T45,
        typename T46>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
        T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41,
        T42, T43, T44, T45, T46, None, None, None, None> {
    typedef Types46<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
            T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38,
            T39, T40, T41, T42, T43, T44, T45, T46>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38, typename T39, typename T40,
        typename T41, typename T42, typename T43, typename T44, typename T45,
        typename T46, typename T47>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
        T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41,
        T42, T43, T44, T45, T46, T47, None, None, None> {
    typedef Types47<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
            T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38,
            T39, T40, T41, T42, T43, T44, T45, T46, T47>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38, typename T39, typename T40,
        typename T41, typename T42, typename T43, typename T44, typename T45,
        typename T46, typename T47, typename T48>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
        T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41,
        T42, T43, T44, T45, T46, T47, T48, None, None> {
    typedef Types48<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
            T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38,
            T39, T40, T41, T42, T43, T44, T45, T46, T47, T48>
            type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38, typename T39, typename T40,
        typename T41, typename T42, typename T43, typename T44, typename T45,
        typename T46, typename T47, typename T48, typename T49>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
        T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
        T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41,
        T42, T43, T44, T45, T46, T47, T48, T49, None> {
    typedef Types49<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
            T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
            T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38,
            T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49>
            type;
};


template <typename T> struct TypeList { typedef Types1<T> type; };

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10,
        typename T11, typename T12, typename T13, typename T14, typename T15,
        typename T16, typename T17, typename T18, typename T19, typename T20,
        typename T21, typename T22, typename T23, typename T24, typename T25,
        typename T26, typename T27, typename T28, typename T29, typename T30,
        typename T31, typename T32, typename T33, typename T34, typename T35,
        typename T36, typename T37, typename T38, typename T39, typename T40,
        typename T41, typename T42, typename T43, typename T44, typename T45,
        typename T46, typename T47, typename T48, typename T49, typename T50>
struct TypeList<
        Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16,
                T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30,
                T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44,
                T45, T46, T47, T48, T49, T50>> {
    typedef typename Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
            T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24,
            T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35,
            T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46,
            T47, T48, T49, T50>::type type;
};

#endif