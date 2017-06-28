#include <shogun/io/TBOutputFormat.h>
#include <shogun/lib/any.h>
#include <tflogger/event.pb.h>
#include <tflogger/summary.pb.h>
#include <utility>
#include <gtest/gtest.h>

using namespace shogun;

#define FORMAT_TEST(type_value, check, value_val) \
void test_case_##type_value() { \
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
void test_case_##type_value() { \
    type_value v = value_val;\
    TBOutputFormat tmp; \
    auto emitted_value = std::make_pair("test", erase_type(v)); \
    std::string node_name="node";\
    EXPECT_THROW(tmp.convert_scalar(1, emitted_value, node_name), ShogunException);\
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

TEST(TBOutputFormat, convert_all_types)
{
    test_case_uint8_t();
    test_case_int16_t();
    test_case_uint16_t();
    test_case_int32_t();
    test_case_uint32_t();
    test_case_int64_t();
    test_case_uint64_t();
    test_case_float32_t();
    test_case_float64_t();
    test_case_floatmax_t();
    test_case_char();
}

TEST(TBOutputFormat, fail_convert)
{
    test_case_complex128_t();
}
