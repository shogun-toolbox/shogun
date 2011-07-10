import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.DNA;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
public class distribution_ppwm_modular {
	static {
		System.loadLibrary("Features");
		System.loadLibrary("Distribution");
	}

	public ArrayList parameter_list = new ArrayList(2);
	public distribution_ppwm_modular() {

		parameter_list.add(Arrays.asList(new Integer(3), new Integer(0)));
		parameter_list.add(Arrays.asList(new Integer(4), new Integer(0)));
	}
	static void run(List para) {
		boolean reverse = false;
		Features.init_shogun_with_defaults();
		int order = ((Integer)para.get(0)).intValue();
		int gap = ((Integer)para.get(1)).intValue();

		String[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");

		StringCharFeatures charfeat = new StringCharFeatures(fm_train_dna, DNA);
		StringWordFeatures feats = new StringWordFeatures(charfeat.get_alphabet());
		feats.obtain_from_char(charfeat, order-1, order, gap, reverse);

		PositionalPWM ppwm = new PositionalPWM();
		ppwm.set_sigma(5.0);
		ppwm.set_mean(10.0);
		DoubleMatrix pwm = new DoubleMatrix(new double[][] {{0.0, 0.5, 0.1, 1.0},
							{0.0, 0.5, 0.5, 0.0},
							{1.0, 0.0, 0.4, 0.0},
							{0.0, 0.0, 0.0, 0.0}});
		//ppwm.set_pwm(DoubleMatrix.log(pwm));
		ppwm.compute_w(20);
		DoubleMatrix w = ppwm.get_w();
	}
	public static void main(String argv[]) {
		distribution_ppwm_modular x = new distribution_ppwm_modular();
		run((List)x.parameter_list.get(0));
	}
}
