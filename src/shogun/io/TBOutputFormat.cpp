/*
* Written (W) 2017 Giovanni De Toni
*/

#include <chrono>
#include <shogun/io/TBOutputFormat.h>
#include <shogun/lib/common.h>

using namespace shogun;

#define CHECK_TYPE(type)\
else if (\
    value.second.type_info().hash_code() == typeid(type).hash_code())\
{\
    summaryValue->set_simple_value(recall_type<type>(value.second));\
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
