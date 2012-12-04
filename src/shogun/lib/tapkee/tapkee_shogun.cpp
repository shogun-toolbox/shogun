#include <shogun/lib/tapkee/tapkee_shogun.hpp>

#define TAPKEE_EIGEN_INCLUDE_FILE <shogun/mathematics/eigen3.h>
#include <shogun/lib/tapkee/tapkee.hpp>
#include <shogun/lib/tapkee/callbacks/pimpl_callbacks.hpp>

TAPKEE_CALLBACK_IS_KERNEL(pimpl_kernel_callback<CKernel>);
TAPKEE_CALLBACK_IS_DISTANCE(pimpl_distance_callback<CDistance>);

using namespace shogun;

class ShogunLoggerImplementation : public LoggerImplementation
{
	virtual void message_info(const std::string& msg)
	{
		SG_SINFO(msg.c_str());
	}
	virtual void message_warning(const std::string& msg)
	{
		SG_SWARNING(msg.c_str());
	}
	virtual void message_error(const std::string& msg)
	{
		SG_SERROR(msg.c_str());
	}
	virtual void message_benchmark(const std::string& msg)
	{
		SG_SINFO(msg.c_str());
	}
};

/*
struct StaticLoggingIntializer
{
public:
	static int initializer;
private:
	static int static_init()
	{
		LoggingSingleton::instance().set_logger_impl(new ShogunLoggerImplementation);
		return 0;
	}
};
int StaticLoggingIntializer::initializer = StaticLoggingIntializer::static_init();
*/

struct shogun_feature_vector_callback
{
	inline void operator()(int i, tapkee::DenseVector& vector) const
	{
	}
};


CDenseFeatures<float64_t>* tapkee_embed(int32_t N, CKernel* kernel, CDistance* distance, CDotFeatures* features)
{
	LoggingSingleton::instance().set_logger_impl(new ShogunLoggerImplementation);

	pimpl_kernel_callback<CKernel> kernel_callback(kernel);
	pimpl_distance_callback<CDistance> distance_callback(distance);
	shogun_feature_vector_callback features_callback;

	tapkee::ParametersMap parameters;
	std::vector<int32_t> indices(N);
	for (size_t i=0; i<static_cast<size_t>(N); i++)
		indices[i] = i;

	tapkee::DenseMatrix result = tapkee::embed(indices.begin(),indices.end(),kernel_callback,distance_callback,features_callback,parameters);
}


