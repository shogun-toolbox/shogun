#include <gtest/gtest.h>
#include "environments/LinearTestEnvironment.h"
#include "environments/MultiLabelTestEnvironment.h"
#include "environments/RegressionTestEnvironment.h"
#include "utils/Utils.h"
#include <shogun/base/some.h>
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
		fs = io::FileSystemRegistry::instance();
		machine = new T();
		SG_REF(machine)

		this->load_data(this->machine->get_machine_problem_type());
	}

	void TearDown()
	{
		SG_UNREF(train_feats)
		SG_UNREF(test_feats)
		SG_UNREF(train_labels)
		SG_UNREF(machine)
		SG_UNREF(deserialized_machine)
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
			SG_SERROR("Unsupported problem type: %d\n", pt);
			FAIL();
		}

		SG_REF(train_feats)
		SG_REF(test_feats)
		SG_REF(train_labels)
	}

	bool serialize_machine(
	    CMachine* cmachine, std::string& filename, bool store_model_features)
	{
		std::string class_name = cmachine->get_name();
		filename = "shogun-unittest-trained-model-serialization-" + class_name +
		           ".XXXXXX";
		generate_temp_filename(const_cast<char*>(filename.c_str()));

		SG_REF(cmachine);
		if (fs->file_exists(filename))
			return false;
		std::unique_ptr<io::WritableFile> file;
		if (fs->new_writable_file(filename, &file))
			return false;
		auto fos = some<io::CFileOutputStream>(file.get());
		auto serializer = some<io::CBitserySerializer>();
		serializer->attach(fos);
		serializer->write(wrap<CSGObject>(cmachine));

		return true;
	}

	bool test_serialization(bool store_model_features = false)
	{
		machine->set_labels(train_labels);
		machine->train(train_feats);

		auto predictions = wrap<CLabels>(machine->apply(test_feats));

		std::string filename;
		if (!serialize_machine(machine, filename, store_model_features))
			return false;

		if (!deserialize_machine(filename))
			return false;

		auto deserialized_predictions =
		    wrap<CLabels>(deserialized_machine->apply(test_feats));

		set_global_fequals_epsilon(1e-7);
		if (!predictions->equals(deserialized_predictions))
			return false;
		set_global_fequals_epsilon(0);
		return true;
	}

	bool deserialize_machine(std::string filename)
	{
		std::unique_ptr<io::RandomAccessFile> raf;
		if (fs->new_random_access_file(filename, &raf))
			return false;
		auto fis = some<io::CFileInputStream>(raf.get());
		auto deserializer = some<io::CBitseryDeserializer>();
		deserializer->attach(fis);
		auto deser_obj = deserializer->read();
		bool delete_success = !fs->delete_file(filename);

		deserialized_machine = dynamic_cast<T*>(deser_obj.get());
		if (deserialized_machine == nullptr)
			return false;
		SG_REF(deserialized_machine);
		return delete_success;
	}

	CDenseFeatures<float64_t> *train_feats, *test_feats;
	CLabels* train_labels;
	T* machine;
	T* deserialized_machine;
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
	CGaussianKernel* kernel = new CGaussianKernel(2.0);
	this->machine->set_kernel(kernel);
	for (auto store_model_features : {false, true})
	{
		EXPECT_TRUE(this->test_serialization(store_model_features));
	}
}
