#include <shogun/features/ArrowFactory.h>
#include <shogun/features/DenseFeatures.h>

#include <algorithm>
#include <vector>

using namespace arrow;
using namespace shogun;
using namespace std;

template<typename ArrowArray, typename T = typename ArrowArray::value_type>
shared_ptr<DenseFeatures<T>> to_dense(const shared_ptr<Table>& table)
{
	auto num_rows = table->num_rows();
	auto num_columns = table->num_columns();
	SG_DEBUG("creating SGMatrix<{}>({}, {}) from arrow::Table", demangled_type<T>(), num_columns, num_rows);
	SGMatrix<T> m(num_columns, num_rows); // we need it transposed 
	for (auto&& c_idx: range(num_columns))
	{
		auto column = table->column(c_idx);
		index_t offset = 0;
		// TODO: this should be parallel
		for (auto&& chunk: range(column->num_chunks()))
		{
			auto array = static_pointer_cast<ArrowArray>(column->chunk(chunk));
			for (index_t j = 0; j < num_rows; ++j)
				m(c_idx,offset+j) = array->Value(j);
			offset += array->length();
		}
	}
	return std::make_shared<DenseFeatures<T>>(m);
}

shared_ptr<Features> shogun::features(const shared_ptr<Table>& table)
{
	auto schema = table->schema();
	auto fields = schema->fields();
	auto num_columns = table->num_columns();

	vector<std::shared_ptr<DataType>> table_types(num_columns);
	transform(fields.begin(), fields.end(), table_types.begin(),
		[](auto f) { return f->type(); });

	bool mixed_type = false;
	auto common_type = table_types.at(0);
	for_each(table_types.begin()+1, table_types.end(),
			[&mixed_type, &common_type](auto t) {
			if (common_type != t)
				mixed_type = false;
	});

	if (!mixed_type)
	{
		switch (common_type->id())
		{
			case Type::BOOL:
				return to_dense<BooleanArray, bool>(table);
			case Type::UINT8:
				return to_dense<UInt8Array>(table);
			case Type::INT8:
				return to_dense<Int8Array>(table);
			case Type::UINT16:
				return to_dense<UInt16Array>(table);
			case Type::INT16:
				return to_dense<Int16Array>(table);
			case Type::UINT32:
				return to_dense<UInt32Array>(table);
			case Type::INT32:
				return to_dense<Int32Array>(table);
			case Type::UINT64:
				return to_dense<UInt64Array>(table);
			case Type::INT64:
				return to_dense<Int64Array>(table);
			case Type::FLOAT:
				return to_dense<FloatArray>(table);
			case Type::DOUBLE:
				return to_dense<DoubleArray>(table);
			default:
				// TODO: add support for all the other types
				throw runtime_error("not supported column type");

		}
	}
	else
	{
		// TODO: combined feature

	}

	return nullptr;
}
