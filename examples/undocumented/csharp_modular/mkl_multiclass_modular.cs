using System;

public class mkl_multiclass_modular {
	public static void Main() {

		modshogun.init_shogun_with_defaults();
		double width = 2.1;
		double epsilon = 1e-5;
		double C = 1.0;
		int mkl_norm = 2;

		double[,] traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		double[,] testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		double[] trainlab = Load.load_labels("../data/label_train_multiclass.dat");

		CombinedKernel kernel = new CombinedKernel();
		CombinedFeatures feats_train = new CombinedFeatures();
		CombinedFeatures feats_test = new CombinedFeatures();

		RealFeatures subkfeats1_train = new RealFeatures(traindata_real);
		RealFeatures subkfeats1_test = new RealFeatures(testdata_real);

		GaussianKernel subkernel = new GaussianKernel(10, width);
		feats_train.append_feature_obj(subkfeats1_train);
		feats_test.append_feature_obj(subkfeats1_test);
		kernel.append_kernel(subkernel);

		RealFeatures subkfeats2_train = new RealFeatures(traindata_real);
		RealFeatures subkfeats2_test = new RealFeatures(testdata_real);

		LinearKernel subkernel2 = new LinearKernel();
		feats_train.append_feature_obj(subkfeats2_train);
		feats_test.append_feature_obj(subkfeats2_test);
		kernel.append_kernel(subkernel2);

		RealFeatures subkfeats3_train = new RealFeatures(traindata_real);
		RealFeatures subkfeats3_test = new RealFeatures(testdata_real);

		PolyKernel subkernel3 = new PolyKernel(10, 2);
		feats_train.append_feature_obj(subkfeats3_train);
		feats_test.append_feature_obj(subkfeats3_test);
		kernel.append_kernel(subkernel3);

		kernel.init(feats_train, feats_train);

		MulticlassLabels labels = new MulticlassLabels(trainlab);

		MKLMulticlass mkl = new MKLMulticlass(C, kernel, labels);
		mkl.set_epsilon(epsilon);
		mkl.set_mkl_epsilon(epsilon);
		mkl.set_mkl_norm(mkl_norm);

		mkl.train();

		kernel.init(feats_train, feats_test);
		double[] outMatrix = LabelsFactory.to_multiclass(mkl.apply()).get_labels();

		modshogun.exit_shogun();
	}
}
