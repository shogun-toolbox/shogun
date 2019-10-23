#include <gtest/gtest.h>
#include "environments/LinearTestEnvironment.h"
#include "environments/MultiLabelTestEnvironment.h"
#include "environments/RegressionTestEnvironment.h"
#include "utils/Utils.h"
#include <shogun/base/ShogunEnv.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/io/CSVFile.h>
#include <shogun/io/SGIO.h>
#include <shogun/io/fs/FileSystem.h>
#include <shogun/io/serialization/BitserySerializer.h>
#include <shogun/io/serialization/BitseryDeserializer.h>
#include <shogun/io/stream/FileInputStream.h>
#include <shogun/io/stream/FileOutputStream.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/machine/Machine.h>

using namespace shogun;
using namespace std;

extern LinearTestEnvironment* linear_test_env;
extern MultiLabelTestEnvironment* multilabel_test_env;
extern RegressionTestEnvironment* regression_test_env;

template <class T>
class TrainedModelSerializationFixture : public ::testing::Test
{
protected:
	void SetUp()
	{
		fs = env();
		machine = std::make_shared<T>();
		this->load_data(this->machine->get_machine_problem_type());
	}

	void TearDown()
	{
		
	}

	void load_data(EProblemType pt)
	{
		switch (pt)
		{
		case PT_BINARY:
		case PT_CLASS:
		{
			std::shared_ptr<GaussianCheckerboard> mock_data =
			    linear_test_env->getBinaryLabelData();
			train_feats = mock_data->get_features_train();
			test_feats = mock_data->get_features_test();
			train_labels = mock_data->get_labels_train();
			break;
		}

		case PT_MULTICLASS:
		{
			std::shared_ptr<GaussianCheckerboard> mock_data =
			    multilabel_test_env->getMulticlassFixture();
			train_feats = mock_data->get_features_train();
			test_feats = mock_data->get_features_test();
			train_labels = mock_data->get_labels_train();
			break;
		}

		case PT_REGRESSION:
			train_feats = regression_test_env->get_features_train();
			test_feats = regression_test_env->get_features_test();
			train_labels = regression_test_env->get_labels_train();
			break;

		default:
			error("Unsupported problem type: {}", pt);
			FAIL();
		}

		
		
		
	}

	bool serialize_machine(
	    const std::shared_ptr<Machine>& cmachine, std::string& filename)
	{
		std::string class_name = cmachine->get_name();
		filename = "shogun-unittest-trained-model-serialization-" + class_name +
		           ".XXXXXX";
		generate_temp_filename(const_cast<char*>(filename.c_str()));

		if (fs->file_exists(filename))
			return false;
		std::unique_ptr<io::WritableFile> file;
		if (fs->new_writable_file(filename, &file))
			return false;
		auto fos = std::make_shared<io::FileOutputStream>(file.get());
		auto serializer = std::make_unique<io::BitserySerializer>();
		serializer->attach(fos);
		serializer->write(cmachine);

		return true;
	}

	bool test_serialization()
	{
		machine->set_labels(train_labels);
		machine->train(train_feats);

		auto predictions = machine->apply(test_feats);

		std::string filename;
		if (!serialize_machine(machine, filename))
			return false;

		if (!deserialize_machine(filename))
			return false;

		auto deserialized_predictions =
		    deserialized_machine->apply(test_feats);

		env()->set_global_fequals_epsilon(1e-7);
		if (!predictions->equals(deserialized_predictions))
			return false;
		env()->set_global_fequals_epsilon(0);
		return true;
	}

	bool deserialize_machine(const std::string& filename)
	{
		std::unique_ptr<io::RandomAccessFile> raf;
		if (fs->new_random_access_file(filename, &raf))
			return false;
		auto fis = std::make_shared<io::FileInputStream>(raf.get());
		auto deserializer = std::make_unique<io::BitseryDeserializer>();
		deserializer->attach(fis);
		auto deser_obj = deserializer->read_object();
		bool delete_success = !fs->delete_file(filename);

		deserialized_machine = deser_obj->as<T>();
		if (deserialized_machine == nullptr)
			return false;
		return delete_success;
	}

	std::shared_ptr<DenseFeatures<float64_t>> train_feats, test_feats;
	std::shared_ptr<Labels> train_labels;
	std::shared_ptr<T> machine;
	std::shared_ptr<T> deserialized_machine;
	io::FileSystemRegistry* fs;
};

#include "trained_model_serialization_test.h"

template <class T>
class TrainedMachineSerialization : public TrainedModelSerializationFixture<T>
{
};

TYPED_TEST_CASE(TrainedMachineSerialization, MachineTypes);

TYPED_TEST(TrainedMachineSerialization, Test)
{
	EXPECT_TRUE(this->test_serialization());
}

template <class T>
class TrainedKernelMachineSerialization
    : public TrainedModelSerializationFixture<T>
{
};

TYPED_TEST_CASE(TrainedKernelMachineSerialization, KernelMachineTypes);

TYPED_TEST(TrainedKernelMachineSerialization, Test)
{
	auto kernel = std::make_shared<GaussianKernel>(2.0);
	this->machine->set_kernel(kernel);
	EXPECT_TRUE(this->test_serialization());
}
