import org.shogun.*;
import org.jblas.*;
import static org.jblas.MatrixFunctions.logi;
import static org.shogun.EAlphabet.DNA;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class distribution_ppwm_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		boolean reverse = false;
		modshogun.init_shogun_with_defaults();
		int order = 3;
		int gap = 4;

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
		ppwm.set_pwm(logi(pwm));
		ppwm.compute_w(20);
		DoubleMatrix w = ppwm.get_w();
		modshogun.exit_shogun();
	}
}
