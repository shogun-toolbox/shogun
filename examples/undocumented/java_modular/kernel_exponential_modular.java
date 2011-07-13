import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.DNA;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class kernel_exponential_modular {
	static {
		System.loadLibrary("Features");
		System.loadLibrary("Kernel");
		System.loadLibrary("Distance");
	}

	public ArrayList parameter_list = new ArrayList(2); 
	public kernel_exponential_modular() {
		parameter_list.add(Arrays.asList(new Double(1.0)));
		parameter_list.add(Arrays.asList(new Double(5.0)));
	}
	public Object run(List para) {
		Features.init_shogun_with_defaults();
		double tau_coef = ((Double)para.get(0)).doubleValue();

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);

		EuclidianDistance distance = new EuclidianDistance(feats_train, feats_train);
		ExponentialKernel kernel= new ExponentialKernel(feats_train, feats_train, tau_coef, distance, 10);

		kernel.init(feats_train, feats_train);
		DoubleMatrix km_train=kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test=kernel.get_kernel_matrix();

		ArrayList result = new ArrayList();
		result.add(km_train);
		result.add(km_test);
		result.add(kernel);

		Features.exit_shogun();
		return (Object)result;
	}
	public static void main(String argv[]) {
		kernel_exponential_modular x = new kernel_exponential_modular();
		x.run((List)x.parameter_list.get(0));
	}
}
