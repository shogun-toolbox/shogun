import org.shogun.*;
import org.jblas.*;
import static org.jblas.DoubleMatrix.randn;

public class evaluation_meansquarederror_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		int N = 100;

		DoubleMatrix ground_truth = randn(1, N);
		DoubleMatrix predicted = randn(1, N);

		RegressionLabels ground_truth_labels = new RegressionLabels(ground_truth);
		RegressionLabels predicted_labels = new RegressionLabels(predicted);

		MeanSquaredError evaluator = new MeanSquaredError();
		double mse = evaluator.evaluate(predicted_labels, ground_truth_labels);

		System.out.println(mse);

		modshogun.exit_shogun();
	}
}
