#include <shogun/io/TBOutputFormat.h>
#include <shogun/lib/any.h>
#include <shogun/lib/tfhistogram/histogram.h>
#include <tflogger/event.pb.h>
#include <tflogger/summary.pb.h>
#include <utility>
#include <vector>
#include <gtest/gtest.h>

using namespace shogun;

#define FORMAT_TEST(type_value, check, value_val) \
void test_case_scalar_##type_value() { \
    type_value v = value_val;\
    tensorflow::Event event_ex; \
    auto summary = event_ex.mutable_summary(); \
	auto summaryValue = summary->add_value(); \
	summaryValue->set_tag("test"); \
	summaryValue->set_node_name("node"); \
    summaryValue->set_simple_value(v); \
    TBOutputFormat tmp; \
    auto emitted_value = std::make_pair("test", erase_type(v)); \
    std::string node_name="node";\
    auto event_gen = tmp.convert_scalar(1, emitted_value, node_name); \
    check(event_gen.summary().value(0).simple_value(), v); \
    EXPECT_EQ(event_gen.summary().value(0).tag(), "test"); \
    EXPECT_EQ(event_gen.summary().value(0).node_name(), "node"); \
}

#define FAILING_FORMAT_TEST(type_value, value_val) \
void test_case_scalar_##type_value() { \
    type_value v = value_val;\
    TBOutputFormat tmp; \
    auto emitted_value = std::make_pair("test", erase_type(v)); \
    std::string node_name="node";\
    EXPECT_THROW(tmp.convert_scalar(1, emitted_value, node_name), ShogunException);\
}

#define FORMAT_TEST_H(type_value) \
void test_case_vector_##type_value() { \
    std::vector<type_value> v = {1, 2, 3, 4};\
    tensorflow::Event event_ex;\
    auto summary = event_ex.mutable_summary();\
	auto summaryValue = summary->add_value();\
	summaryValue->set_tag("test");\
	summaryValue->set_node_name("node");\
\
    tensorflow::histogram::Histogram h;\
    tensorflow::HistogramProto * hp = new tensorflow::HistogramProto();\
    for (auto value_v : v)\
        h.Add(value_v);\
    h.EncodeToProto(hp, true);\
    summaryValue->set_allocated_histo(hp);\
\
    TBOutputFormat tmp;\
    auto emitted_value = std::make_pair("test", erase_type(v));\
    std::string node_name="node";\
    auto event_gen = tmp.convert_vector(1, emitted_value, node_name);\
\
    tensorflow::histogram::Histogram h2;\
    h2.DecodeFromProto(event_gen.summary().value(0).histo());\
    \
    EXPECT_EQ(h2.ToString(), h.ToString());\
    EXPECT_EQ(event_gen.summary().value(0).tag(), "test");\
    EXPECT_EQ(event_gen.summary().value(0).node_name(), "node");\
}

#define FAILING_FORMAT_TEST_H(type_value) \
void test_case_vector_##type_value() { \
    std::vector<type_value> v = {1, 2, 3, 4};\
    TBOutputFormat tmp; \
    auto emitted_value = std::make_pair("test", erase_type(v)); \
    std::string node_name="node";\
    EXPECT_THROW(tmp.convert_vector(1, emitted_value, node_name), ShogunException);\
}

FORMAT_TEST(uint8_t, EXPECT_EQ, 1)
FORMAT_TEST(int16_t, EXPECT_EQ, 1)
FORMAT_TEST(uint16_t, EXPECT_EQ, 1)
FORMAT_TEST(int32_t, EXPECT_EQ, 1)
FORMAT_TEST(uint32_t, EXPECT_EQ, 1)
FORMAT_TEST(int64_t, EXPECT_EQ, 1)
FORMAT_TEST(uint64_t, EXPECT_EQ, 1)
FORMAT_TEST(float32_t, EXPECT_FLOAT_EQ, 1)
FORMAT_TEST(float64_t, EXPECT_FLOAT_EQ, 1)
FORMAT_TEST(floatmax_t, EXPECT_FLOAT_EQ, 1)
FORMAT_TEST(char, EXPECT_EQ, 1)
FAILING_FORMAT_TEST(complex128_t, 0)

FORMAT_TEST_H(uint8_t)
FORMAT_TEST_H(int16_t)
FORMAT_TEST_H(uint16_t)
FORMAT_TEST_H(int32_t)
FORMAT_TEST_H(uint32_t)
FORMAT_TEST_H(int64_t)
FORMAT_TEST_H(uint64_t)
FORMAT_TEST_H(float32_t)
FORMAT_TEST_H(float64_t)
FORMAT_TEST_H(floatmax_t)
FORMAT_TEST_H(char)
FAILING_FORMAT_TEST_H(complex128_t)

TEST(TBOutputFormat, convert_all_types_scalar)
{
    test_case_scalar_uint8_t();
    test_case_scalar_int16_t();
    test_case_scalar_uint16_t();
    test_case_scalar_int32_t();
    test_case_scalar_uint32_t();
    test_case_scalar_int64_t();
    test_case_scalar_uint64_t();
    test_case_scalar_float32_t();
    test_case_scalar_float64_t();
    test_case_scalar_floatmax_t();
    test_case_scalar_char();
}

TEST(TBOutputFormat, fail_convert)
{
    test_case_scalar_complex128_t();
}

TEST(TBOutputFormat, convert_all_types_histo)
{
    test_case_vector_uint8_t();
    test_case_vector_int16_t();
    test_case_vector_uint16_t();
    test_case_vector_int32_t();
    test_case_vector_uint32_t();
    test_case_vector_int64_t();
    test_case_vector_uint64_t();
    test_case_vector_float32_t();
    test_case_vector_float64_t();
    test_case_vector_floatmax_t();
    test_case_vector_char();
}

TEST(TBOutputFormat, fail_convert_histo)
{
    test_case_vector_complex128_t();
}
