import org.shogun.*;
import org.jblas.*;
import static org.jblas.DoubleMatrix.randn;

public class evaluation_meansquarederror_modular {
	static {
		System.loadLibrary("Features");
		System.loadLibrary("Evaluation");
	}

	public static void main(String argv[]) {
		Features.init_shogun_with_defaults();
		int N = 100;		

		DoubleMatrix ground_truth = randn(1, N);
		DoubleMatrix predicted = randn(1, N);

		Labels ground_truth_labels = new Labels(ground_truth);
		Labels predicted_labels = new Labels(predicted);

		MeanSquaredError evaluator = new MeanSquaredError();
		double mse = evaluator.evaluate(predicted_labels, ground_truth_labels);

		System.out.println(mse);

		Features.exit_shogun();
	}
}
