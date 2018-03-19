#include <gtest/gtest.h>
#include <shogun/features/ArrowFactory.h>
#include <shogun/features/DenseFeatures.h>
#include <arrow/memory_pool.h>
#include <random>

using namespace shogun;
using namespace arrow;

#define ARROW_EXPECT_OK(expr)         \
  do {                          \
    ::arrow::Status s = (expr); \
    EXPECT_TRUE(s.ok());        \
} while (false)

static inline void random_bytes(int64_t n, uint32_t seed, uint8_t* out)
{
  std::mt19937 gen(seed);
  std::uniform_int_distribution<uint32_t> d(0, std::numeric_limits<uint8_t>::max());
  std::generate(out, out + n, [&d, &gen] { return static_cast<uint8_t>(d(gen)); });
}

template<typename T>
class ArrowTest : public ::testing::Test
{
public:
	void SetUp()
	{
		pool_ = ::arrow::default_memory_pool();
		random_seed_ = 0;

		auto f0 = field("f0", std::make_shared<T>());
		auto f1 = field("f1", std::make_shared<T>());
		auto schema = arrow::schema({f0, f1});
		std::vector<std::shared_ptr<Array>> arrays = {
			MakeRandomArray(column_length_),
			MakeRandomArray(column_length_)};
		auto columns = {
			std::make_shared<ChunkedArray>(arrays[0]),
			std::make_shared<ChunkedArray>(arrays[1])};
		table_ = Table::Make(schema, columns);
	}

	std::shared_ptr<::arrow::Buffer> MakeRandomNullBitmap(int64_t length, int64_t null_count) {
		const int64_t null_nbytes = BitUtil::BytesForBits(length);

		std::shared_ptr<Buffer> null_bitmap;
		ARROW_EXPECT_OK(AllocateBuffer(pool_, null_nbytes, &null_bitmap));
		memset(null_bitmap->mutable_data(), 255, null_nbytes);
		for (int64_t i = 0; i < null_count; i++) {
			BitUtil::ClearBit(null_bitmap->mutable_data(), i * (length / null_count));
		}
		return null_bitmap;

	}

	std::shared_ptr<::arrow::Array> MakeRandomArray(int64_t length, int64_t null_count = 0)
	{
		const int64_t data_nbytes = length * sizeof(typename T::c_type);
		std::shared_ptr<Buffer> data;
		ARROW_EXPECT_OK(AllocateBuffer(pool_, data_nbytes, &data));

		// Fill with random data
		random_bytes(data_nbytes, random_seed_++, data->mutable_data());
		std::shared_ptr<Buffer> null_bitmap = MakeRandomNullBitmap(length, null_count);

		return std::make_shared<NumericArray<T>>(length, data, null_bitmap, null_count);
	}

	auto table() const
	{
		return table_;
	}

	int64_t column_length_ = 10;
 protected:
	uint32_t random_seed_;
	::arrow::MemoryPool* pool_;
	std::shared_ptr<Table> table_;
};


using ArrayNumericTypes = ::testing::Types<
	UInt8Type, Int8Type, UInt16Type, Int16Type, UInt32Type,
	Int32Type, UInt64Type, Int64Type, FloatType, DoubleType>;
TYPED_TEST_CASE(ArrowTest, ArrayNumericTypes);

TYPED_TEST(ArrowTest, test_dense)
{
	auto table = this->table();
	auto shogun_features = features(table);
	auto dense_features = std::dynamic_pointer_cast<DenseFeatures<typename TypeParam::c_type>>(shogun_features);
	ASSERT_TRUE(dense_features != nullptr);
	ASSERT_EQ(this->column_length_, dense_features->get_num_vectors());
	ASSERT_EQ(2, dense_features->get_num_features());

	auto matrix = dense_features->get_feature_matrix();
	for (index_t i = 0; i < matrix.num_rows; ++i)
	{
		auto da = std::static_pointer_cast<NumericArray<TypeParam>>(table->column(i)->chunk(0));
		for (index_t j = 0; j < matrix.num_cols; ++j)
		{
			EXPECT_DOUBLE_EQ(da->Value(j), matrix(i, j));
		}
	}
}

