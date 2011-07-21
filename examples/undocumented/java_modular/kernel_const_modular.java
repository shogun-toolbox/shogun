import org.shogun.*;
import org.jblas.*;

public class kernel_const_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		double c = 23;

		DummyFeatures feats_train = new DummyFeatures(10);
		DummyFeatures feats_test = new DummyFeatures(17);

		ConstKernel kernel = new ConstKernel(feats_train, feats_train, c);

		DoubleMatrix km_train = kernel.get_kernel_matrix();

		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test=kernel.get_kernel_matrix();

		System.out.println(km_train.toString());
		System.out.println(km_test.toString());

		modshogun.exit_shogun();
	}
}
