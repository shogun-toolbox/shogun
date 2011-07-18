import org.shogun.*;
import org.jblas.*;
public class mkl_multiclass_modular {
	static {
		System.loadLibrary("Features");
		System.loadLibrary("Classifier");
		System.loadLibrary("Kernel");
	}

	public static void main(String argv[]) {
		Features.init_shogun_with_defaults();
		double width = 2.1;
		double epsilon = 1e-5;
		double C = 1.0;
		int mkl_norm = 2;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		DoubleMatrix trainlab = Load.load_labels("../data/label_train_twoclass.dat");

		CombinedKernel kernel = new CombinedKernel();
		CombinedFeatures feats_train = new CombinedFeatures();
		CombinedFeatures feats_test = new CombinedFeatures();

		RealFeatures subkfeats_train = new RealFeatures(traindata_real);
		RealFeatures subkfeats_test = new RealFeatures(testdata_real);
		
		GaussianKernel subkernel = new GaussianKernel(10, width);
		feats_train.append_feature_obj(subkfeats_train);
		feats_test.append_feature_obj(subkfeats_test);
		kernel.append_kernel(subkernel);

		LinearKernel subkernel2 = new LinearKernel();
		feats_train.append_feature_obj(subkfeats_train);
		feats_test.append_feature_obj(subkfeats_test);
		kernel.append_kernel(subkernel2);

		PolyKernel subkernel3 = new PolyKernel(10, 2);
		feats_train.append_feature_obj(subkfeats_train);
		feats_test.append_feature_obj(subkfeats_test);
		kernel.append_kernel(subkernel3);
		

		kernel.init(feats_train, feats_train);

		Labels labels = new Labels(trainlab);

		MKLMultiClass mkl = new MKLMultiClass(C, kernel, labels);
		mkl.set_epsilon(epsilon);
		mkl.set_mkl_epsilon(epsilon);
		mkl.set_mkl_norm(mkl_norm);

		mkl.train();

		kernel.init(feats_train, feats_test);
		DoubleMatrix out =  mkl.apply().get_labels();

		Features.exit_shogun();
	}
}
