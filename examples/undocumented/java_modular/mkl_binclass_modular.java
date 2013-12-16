import org.shogun.*;
import org.jblas.*;

public class mkl_binclass_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		double width = 2.1;
		double epsilon = 1e-5;
		double C = 1.0;
		int mkl_norm = 2;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		DoubleMatrix trainlab = Load.load_labels("../data/label_train_twoclass.dat");

		RealFeatures tfeats = new RealFeatures(traindata_real);
		PolyKernel tkernel = new PolyKernel(10,3);
		tkernel.init(tfeats, tfeats);
	DoubleMatrix K_train = tkernel.get_kernel_matrix();

		RealFeatures pfeats = new RealFeatures(testdata_real);
		tkernel.init(tfeats, pfeats);
	DoubleMatrix K_test = tkernel.get_kernel_matrix();

		CombinedFeatures feats_train = new CombinedFeatures();
	feats_train.append_feature_obj(new RealFeatures(traindata_real));

		CombinedKernel kernel = new CombinedKernel();
		kernel.append_kernel(new CustomKernel(K_train));
		kernel.append_kernel(new PolyKernel(10,2));
		kernel.init(feats_train, feats_train);

		BinaryLabels labels = new BinaryLabels(trainlab);

		MKLClassification mkl = new MKLClassification();
		mkl.set_mkl_norm(1);
		mkl.set_kernel(kernel);
	mkl.set_labels(labels);

		mkl.train();

		CombinedFeatures feats_pred = new CombinedFeatures();
	feats_pred.append_feature_obj(new RealFeatures(testdata_real));

		CombinedKernel kernel2 = new CombinedKernel();
		kernel2.append_kernel(new CustomKernel(K_test));
		kernel2.append_kernel(new PolyKernel(10, 2));
		kernel2.init(feats_train, feats_pred);

		mkl.set_kernel(kernel2);
		mkl.apply();

	}
}
