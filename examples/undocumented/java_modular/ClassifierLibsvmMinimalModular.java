import org.shogun.*;
import org.jblas.*;
public class ClassifierLibsvmMinimalModular {
	static {
		System.loadLibrary("Features");
		System.loadLibrary("Classifier");
		System.loadLibrary("Kernel");
	}
	
	public static double mean_error(DoubleMatrix out, DoubleMatrix lab) {
		out.assertSameLength(lab);
		double mean=0;
		for (int i=0; i<lab.length; i++) {
			if (java.lang.Math.signum(out.get(i)) != java.lang.Math.signum(lab.get(i)))
				mean++;
		}
		mean/=lab.length;
		return mean;
	}

	
	public static void main(String argv[]) {
		Features.init_shogun_with_defaults();
		int num = 1000;
		double dist = 1.0;
		double width = 2.1;
		double C = 1.0;

		DoubleMatrix offs=DoubleMatrix.ones(2, num).mmul(dist);
		DoubleMatrix x = DoubleMatrix.randn(2, num).sub(offs);
		DoubleMatrix y = DoubleMatrix.randn(2, num).add(offs);
		DoubleMatrix traindata_real = DoubleMatrix.concatHorizontally(x, y);

		DoubleMatrix m = DoubleMatrix.randn(2, num).sub(offs);
		DoubleMatrix n = DoubleMatrix.randn(2, num).add(offs);
		DoubleMatrix testdata_real = DoubleMatrix.concatHorizontally(m, n);

		DoubleMatrix trainlab = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(1, num).neg(),
				DoubleMatrix.ones(1, num));
		DoubleMatrix testlab = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(1, num).neg(),
				DoubleMatrix.ones(1, num));

		RealFeatures feats_train = new RealFeatures();
		feats_train.set_feature_matrix(traindata_real);
		RealFeatures feats_test = new RealFeatures();
		feats_test.set_feature_matrix(testdata_real);
		GaussianKernel kernel = new GaussianKernel(feats_train, feats_train, width);
		Labels labels = new Labels(trainlab);
		LibSVM svm = new LibSVM(C, kernel, labels);
		svm.train();

		kernel.init(feats_train, feats_test);
		DoubleMatrix out = svm.apply().get_labels();

		double mean=mean_error(out, testlab);
		System.out.println("Mean Error = " + mean);
		Features.exit_shogun();
	}
}
