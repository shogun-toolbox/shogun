import org.shogun.*;
import org.jblas.*;
public class kernel_combined_custom_poly_modular {
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

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		DoubleMatrix trainlab = Load.load_labels("../data/label_train_twoclass.dat");

		CombinedKernel kernel = new CombinedKernel();
		CombinedFeatures feats_train = new CombinedFeatures();

		RealFeatures tfeats = new RealFeatures(traindata_real);
		PolyKernel tkernel = new PolyKernel(10,3);
		tkernel.init(tfeats, tfeats);
		DoubleMatrix K = tkernel.get_kernel_matrix();
		kernel.append_kernel(new CustomKernel(K));

		RealFeatures subkfeats_train = new RealFeatures(traindata_real);
		feats_train.append_feature_obj(subkfeats_train);
		PolyKernel subkernel = new PolyKernel(10,2);
		kernel.append_kernel(subkernel);

		kernel.init(feats_train, feats_train);

		Labels labels = new Labels(trainlab);

		LibSVM svm = new LibSVM(C, kernel, labels);
		svm.train();

		CombinedKernel kernel_pred = new CombinedKernel();
		CombinedFeatures feats_pred = new CombinedFeatures();

		RealFeatures pfeats = new RealFeatures(testdata_real);
		PolyKernel tkernel_pred = new PolyKernel(10,3);
		tkernel_pred.init(tfeats, pfeats);
		DoubleMatrix KK = tkernel.get_kernel_matrix();
		kernel_pred.append_kernel(new CustomKernel(KK));

		RealFeatures subkfeats_test = new RealFeatures(testdata_real);
		feats_pred.append_feature_obj(subkfeats_train);
		PolyKernel subkernel_pred = new PolyKernel(10,2);
		kernel_pred.append_kernel(subkernel_pred);

		kernel_pred.init(feats_train, feats_pred);

		svm.set_kernel(kernel_pred);
		svm.apply();
		DoubleMatrix km_train=kernel.get_kernel_matrix();
		System.out.println(km_train.toString());

		Features.exit_shogun();
	}
}
