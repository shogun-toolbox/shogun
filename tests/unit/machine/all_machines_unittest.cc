/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann
 */

#include <gtest/gtest.h>

#include "utils/SGObjectIterator.h"
#include "utils/Utils.h"

#include <shogun/io/fs/FileSystem.h>
#include <shogun/io/serialization/BitserySerializer.h>
#include <shogun/io/serialization/BitseryDeserializer.h>
#include <shogun/io/stream/FileInputStream.h>
#include <shogun/io/stream/FileOutputStream.h>

#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/CrossValidationSplitting.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/evaluation/MeanSquaredError.h>

#include <shogun/kernel/GaussianKernel.h>
#include <shogun/distance/EuclideanDistance.h>

#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/classifier/Perceptron.h>
#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/multiclass/MulticlassLibLinear.h>
#include <shogun/multiclass/KNN.h>
#include <shogun/regression/LinearRidgeRegression.h>
#include <shogun/regression/KernelRidgeRegression.h>
#include <shogun/regression/svr/LibLinearRegression.h>
#include <shogun/machine/RandomForest.h>

#include <shogun/mathematics/NormalDistribution.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

#include <shogun/lib/View.h>


using namespace shogun;
using namespace std;

void init_machine(CMachine* machine)
{
	if (auto* casted = dynamic_cast<CKernelMachine*>(machine))
		casted->set_kernel(new CGaussianKernel());
	if (auto* casted = dynamic_cast<CDistanceMachine*>(machine))
		casted->set_distance(new CEuclideanDistance());

	if (machine->has("max_iterations"))
		machine->put("max_iterations", 50);
}

CCrossValidation*  generate_cv(CMachine* machine, const std::pair<CDenseFeatures<float64_t>*, CDenseLabels*>& data)
{
	auto ss = new CCrossValidationSplitting(data.second, 5);
	CEvaluation* ec = nullptr;
	switch (machine->get_machine_problem_type())
	{
	case PT_BINARY:
		ec = new CAccuracyMeasure();
		break;
	case PT_MULTICLASS:
		ec = new CMulticlassAccuracy();
		break;
	case PT_REGRESSION:
		ec = new CMeanSquaredError();
		break;
	default:
		SG_SNOTIMPLEMENTED
		break;
	}

	auto cv = new CCrossValidation(machine, data.first, data.second, ss, ec);
	cv->set_num_runs(3);
	SG_REF(cv);

	return cv;
}

std::pair<CDenseFeatures<float64_t>*, CDenseLabels*> generate_data(const CMachine* machine)
{
	auto N = 50;
	auto D = 3;

	std::mt19937_64 prng(57);
	NormalDistribution<float64_t> randn;
	UniformRealDistribution<float64_t> rand(0,1);
	UniformIntDistribution<int32_t> randi(0,2);

	SGMatrix<float64_t> X(D,N);
	for (auto i : range(D*N))
		X.matrix[i] = randn(prng);
	auto features = new CDenseFeatures<float64_t>(X);

	SGVector<float64_t> y_reg(N);
	SGVector<float64_t> y_binary(N);
	SGVector<float64_t> y_mc(N);

	for (auto i : range(N))
	{
		auto redux = linalg::mean(X.get_column(i));

		y_reg[i] = redux + std::sin(redux) + 1;
		y_mc[i] = redux<0 ? 0 : 1;
		y_binary[i] = y_mc[i] * 2 -1;

		// noise
		y_reg[i] +=  randn(prng)*0.1;
		if (rand(prng)>0.1)
		{
			y_binary[i] *= (-1);
			y_mc[i] = (int32_t(y_mc[i]) + randi(prng)) % 3;
		}
	}

	CDenseLabels* labels;
	auto pt = machine->get_machine_problem_type();
	switch (pt)
	{
	case PT_BINARY:
	case PT_CLASS:
	{
		labels = new CBinaryLabels(y_binary);
		break;
	}

	case PT_MULTICLASS:
	{
		labels = new CMulticlassLabels(y_mc);
		break;
	}

	case PT_REGRESSION:
		labels = new CRegressionLabels(y_reg);
		break;

	default:
		SG_SERROR("Unsupported problem type: %d\n", pt);
	}

	SG_REF(features)
	SG_REF(labels)
	return make_pair(features, labels);
}

void serialize_machine(CMachine* machine, std::string& fname)
{
	auto fs = io::FileSystemRegistry::instance();
	std::string class_name = machine->get_name();
	fname = "shogun-unittest-AllMachines-trained_model_seiralization_consistency-" + class_name +
			   ".XXXXXX";
	generate_temp_filename(const_cast<char*>(fname.c_str()));

	SG_REF(machine);
	EXPECT_FALSE(fs->file_exists(fname));
	std::unique_ptr<io::WritableFile> file;
	EXPECT_FALSE(fs->new_writable_file(fname, &file));
	auto fos = some<io::CFileOutputStream>(file.get());
	auto serializer = some<io::CBitserySerializer>();
	serializer->attach(fos);
	serializer->write(wrap<CSGObject>(machine));
}

CMachine* deserialize_machine(std::string fname)
{
	auto fs = io::FileSystemRegistry::instance();
	std::unique_ptr<io::RandomAccessFile> raf;
	EXPECT_FALSE(fs->new_random_access_file(fname, &raf));
	auto fis = some<io::CFileInputStream>(raf.get());
	auto deserializer = some<io::CBitseryDeserializer>();
	deserializer->attach(fis);
	auto deser_obj = deserializer->read_object();
	bool delete_success = !fs->delete_file(fname);
	EXPECT_TRUE(delete_success);

	return dynamic_cast<CMachine*>(deser_obj.get());
}

// TODO, generate this automatically, like in trained_model_serialization
std::set<string> all_machines = {"LibSVM", "Perceptron", "LibLinear",
		"MulticlassLibLinear", "LinearRidgeRegression", "KNN",
		"KernelRidgeRegression", "LibLinearRegression", "RandomForest"};

TEST(AllMachines, train_uninitialized)
{
	std::set<std::string> ignores = {};
	for (auto obj : sg_object_iterator<untemplated_sgobject>(all_machines).ignore(ignores))
	{
		auto machine = obj->as<CMachine>();
		SCOPED_TRACE(machine->get_name());

		EXPECT_THROW(machine->train(), ShogunException);
	}
}

TEST(AllMachines, train_execute)
{
	std::set<std::string> ignores = {};
	for (auto obj : sg_object_iterator<untemplated_sgobject>(all_machines).ignore(ignores))
	{
		auto machine = obj->as<CMachine>();
		SCOPED_TRACE(machine->get_name());

		init_machine(machine);
		auto data = generate_data(machine);
		machine->set_labels(data.second);
		machine->train(data.first);
	}
}

TEST(AllMachines, train_thread_consistency)
{
	std::set<std::string> ignores = {
			"RandomForest" // segfault
			};
	for (auto obj : sg_object_iterator<untemplated_sgobject>(all_machines).ignore(ignores))
	{
		auto machine = obj->as<CMachine>();
		SCOPED_TRACE(machine->get_name());

		init_machine(machine);
		auto machine2 = make_clone(machine);

		auto data = generate_data(machine);

		get_global_parallel()->set_num_threads(1);
		machine->set_labels(data.second);
		if (machine->has("seed"))
			machine->put("seed", 1);
		machine->train(data.first);
		auto result_single = machine->apply(data.first);

		init_machine(machine);
		get_global_parallel()->set_num_threads(4);
		machine2->set_labels(data.second);
		if (machine2->has("seed"))
			machine2->put("seed", 1);
		machine2->train(data.first);
		auto result_multi = machine2->apply(data.first);

		EXPECT_TRUE(result_single->equals(result_multi));
	}
}


TEST(AllMachines, view_subsampling_consistency)
{
	std::set<std::string> ignores = {
		"RandomForest" // segfault
	};
	for (auto obj : sg_object_iterator<untemplated_sgobject>(all_machines).ignore(ignores))
	{
		auto machine = obj->as<CMachine>();
		SCOPED_TRACE(machine->get_name());

		init_machine(machine);
		auto data = generate_data(machine);

		auto X = data.first->get_feature_matrix();
		auto y = data.second->get_labels();

		SGVector<index_t> subset = {1,3,4,6};

		auto features_subset = view(data.first, subset);
		auto labels_subset = view(data.second, subset);

		SGMatrix<float64_t> X_subsampled(X.num_rows, subset.size());
		SGVector<float64_t> y_subsampled(subset.size());

		for (auto i : range(subset.size()))
		{
			memcpy(X_subsampled.get_column_vector(i), X.get_column_vector(subset[i]), X.num_rows * sizeof(decltype(X(0,0))));
			y_subsampled[i] = y[subset[i]];
		}

		auto features_subsampled = new CDenseFeatures<float64_t>(X_subsampled);
		auto labels_subsampled = make_clone(data.second);
		labels_subsampled->set_labels(y_subsampled);

		auto machine_subset = make_clone(machine);
		auto machine_subsampled = make_clone(machine);
		if (machine->has("seed"))
		{
			machine_subset->put("seed", 1);
			machine_subsampled->put("seed", 1);
		}

		machine_subset->set_labels(labels_subset);
		machine_subset->train(features_subset);
		machine_subsampled->set_labels(labels_subsampled);
		machine_subsampled->train(features_subsampled);

		auto result_subset = machine_subset->apply(data.first);
		auto result_subsampled = machine_subsampled->apply(data.first);

		EXPECT_TRUE(result_subset->equals(result_subsampled));
	}
}

TEST(AllMachines, cv_thread_consistency)
{
	std::set<std::string> ignores = {
			"RandomForest" // segfault
			};
	for (auto obj : sg_object_iterator<untemplated_sgobject>(all_machines).ignore(ignores))
	{
		auto machine = obj->as<CMachine>();
		SCOPED_TRACE(machine->get_name());

		init_machine(machine);
		auto data = generate_data(machine);
		auto cv = generate_cv(machine, data);
		auto cv2 = make_clone(cv);

		get_global_parallel()->set_num_threads(1);
		cv->put("seed", 1);
		auto result_single = cv->evaluate();

		get_global_parallel()->set_num_threads(4);
		cv2->put("seed", 1);
		auto result_multi = cv2->evaluate();

		EXPECT_TRUE(result_single->equals(result_multi));
	}
}

TEST(AllMachines, train_apply_no_side_effects)
{
	std::set<std::string> ignores = {
			"RandomForest" // segfault
	};
	for (auto obj : sg_object_iterator<untemplated_sgobject>(all_machines).ignore(ignores))
	{
		auto machine = obj->as<CMachine>();
		SCOPED_TRACE(machine->get_name());

		init_machine(machine);
		auto data = generate_data(machine);

		auto features_before = data.first->clone();
		auto labels_before = data.second->clone();

		machine->set_labels(data.second);
		machine->train(data.first);
		machine->apply(data.first);

		auto features_after = data.first->clone();
		auto labels_after = data.second->clone();

		EXPECT_TRUE(features_before->equals(features_after));
		EXPECT_TRUE(labels_before->equals(labels_after));
	}
}

TEST(AllMachines, trained_model_serialization_consistency)
{
	std::set<std::string> ignores = {
			};
	for (auto obj : sg_object_iterator<untemplated_sgobject>(all_machines).ignore(ignores))
	{
		auto machine = obj->as<CMachine>();
		SCOPED_TRACE(machine->get_name());

		init_machine(machine);
		auto data = generate_data(machine);

		machine->set_labels(data.second);
		machine->train(data.first);

		auto predictions = machine->apply(data.first);

		std::string filename;
		serialize_machine(machine, filename);
		auto deserialized_machine = deserialize_machine(filename);

		auto deserialized_predictions = deserialized_machine->apply(data.first);

		EXPECT_TRUE(predictions->equals(deserialized_predictions));
	}
}
