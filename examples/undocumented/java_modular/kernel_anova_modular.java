import org.shogun.*;
import org.jblas.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class kernel_anova_modular implements test {
	static {
		System.loadLibrary("Features");
		System.loadLibrary("Kernel");
	}

	public ArrayList parameter_list = new ArrayList(2); 
	public kernel_anova_modular() {
		parameter_list.add(Arrays.asList(new Integer(2), new Integer(10)));
		parameter_list.add(Arrays.asList(new Integer(5), new Integer(10)));
	}
	public Object run(List para) {
		Features.init_shogun_with_defaults();
		int cardinality = ((Integer)para.get(0)).intValue();
		int size_cache = ((Integer)para.get(1)).intValue();

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);

		ANOVAKernel kernel = new ANOVAKernel(feats_train, feats_train, cardinality, size_cache);

		DoubleMatrix km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test = kernel.get_kernel_matrix();

		ArrayList result = new ArrayList();
		result.add(km_train);
		result.add(km_test);
		result.add(kernel);

		Features.exit_shogun();
		return (Object)result;
	}
	public static void main(String argv[]) {
		kernel_anova_modular x = new kernel_anova_modular();
		x.run((List)x.parameter_list.get(0));
	}
}
