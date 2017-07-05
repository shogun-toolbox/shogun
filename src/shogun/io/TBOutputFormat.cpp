/*
* Written (W) 2017 Giovanni De Toni
*/

#include <chrono>
#include <vector>
#include <shogun/io/TBOutputFormat.h>
#include <shogun/lib/common.h>
#include <shogun/lib/tfhistogram/histogram.h>

using namespace shogun;

#define CHECK_TYPE(type)\
else if (\
    value.second.type_info().hash_code() == typeid(type).hash_code())\
{\
    summaryValue->set_simple_value(recall_type<type>(value.second));\
}

#define CHECK_TYPE_HISTO(type)\
else if (\
    value.second.type_info().hash_code() == typeid(type).hash_code())\
{\
    tensorflow::histogram::Histogram h;\
    tensorflow::HistogramProto * hp = new tensorflow::HistogramProto();\
    auto v = recall_type<type>(value.second);\
    for (auto value_v : v)\
        h.Add(value_v);\
    h.EncodeToProto(hp, true);\
    summaryValue->set_allocated_histo(hp);\
}

TBOutputFormat::TBOutputFormat(){};

TBOutputFormat::~TBOutputFormat(){};

tensorflow::Event TBOutputFormat::convert_scalar(
    const int64_t& event_step, const std::pair<std::string, Any>& value,
    std::string& node_name)
{
	auto millisec = std::chrono::duration_cast<std::chrono::milliseconds>(
	                    std::chrono::system_clock::now().time_since_epoch())
	                    .count();

	tensorflow::Event e;
	e.set_wall_time(millisec);
	e.set_step(event_step);

	tensorflow::Summary* summary = e.mutable_summary();
	auto summaryValue = summary->add_value();
	summaryValue->set_tag(value.first);
	summaryValue->set_node_name(node_name);

	if (value.second.type_info().hash_code() == typeid(int8_t).hash_code())
	{
		summaryValue->set_simple_value(recall_type<int8_t>(value.second));
	}
    CHECK_TYPE(uint8_t)
    CHECK_TYPE(int16_t)
    CHECK_TYPE(uint16_t)
    CHECK_TYPE(int32_t)
    CHECK_TYPE(uint32_t)
    CHECK_TYPE(int64_t)
    CHECK_TYPE(uint64_t)
    CHECK_TYPE(float32_t)
    CHECK_TYPE(float64_t)
    CHECK_TYPE(floatmax_t)
    CHECK_TYPE(char)
	else {
        SG_ERROR("Unsupported type %s", value.second.type_info().name());
    }

	return e;
}

tensorflow::Event TBOutputFormat::convert_vector(
    const int64_t& event_step, const std::pair<std::string, Any>& value,
    std::string& node_name)
{
    auto millisec = std::chrono::duration_cast<std::chrono::milliseconds>(
	                    std::chrono::system_clock::now().time_since_epoch())
	                    .count();

	tensorflow::Event e;
	e.set_wall_time(millisec);
	e.set_step(event_step);

	tensorflow::Summary* summary = e.mutable_summary();
	auto summaryValue = summary->add_value();
	summaryValue->set_tag(value.first);
	summaryValue->set_node_name(node_name);

	if (value.second.type_info().hash_code() == typeid(std::vector<int8_t>).hash_code())
	{
        tensorflow::histogram::Histogram h;
        tensorflow::HistogramProto * hp = new tensorflow::HistogramProto();
        auto v = recall_type<std::vector<int8_t>>(value.second);
        for (auto value_v : v)
            h.Add(value_v);
        h.EncodeToProto(hp, true);
        summaryValue->set_allocated_histo(hp);
	}
    CHECK_TYPE_HISTO(std::vector<uint8_t>)
    CHECK_TYPE_HISTO(std::vector<int16_t>)
    CHECK_TYPE_HISTO(std::vector<uint16_t>)
    CHECK_TYPE_HISTO(std::vector<int32_t>)
    CHECK_TYPE_HISTO(std::vector<uint32_t>)
    CHECK_TYPE_HISTO(std::vector<int64_t>)
    CHECK_TYPE_HISTO(std::vector<uint64_t>)
    CHECK_TYPE_HISTO(std::vector<float32_t>)
    CHECK_TYPE_HISTO(std::vector<float64_t>)
    CHECK_TYPE_HISTO(std::vector<floatmax_t>)
    CHECK_TYPE_HISTO(std::vector<char>)
	else {
        SG_ERROR("Unsupported type %s", value.second.type_info().name());
    }

	return e;
}
