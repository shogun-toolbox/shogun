import org.shogun.*;
import org.jblas.*;
import static org.jblas.DoubleMatrix.randn;

public class kernel_auc_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		double width = 1.6;

		DoubleMatrix train_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix trainlab = Load.load_labels("../data/label_train_twoclass.dat");

		RealFeatures feats_train = new RealFeatures(train_real);
		GaussianKernel subkernel = new GaussianKernel(feats_train, feats_train, width);


		BinaryLabels labels = new BinaryLabels(trainlab);

		AUCKernel kernel = new AUCKernel(0, subkernel);
		kernel.setup_auc_maximization(labels);

		DoubleMatrix km_train = kernel.get_kernel_matrix();
		System.out.println(km_train.toString());

	}
}
