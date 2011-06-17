import org.shogun.*;
import org.jblas.*;
import static org.jblas.DoubleMatrix.randn;

public class evaluation_multiclassaccuracy_modular {
	static {
		System.loadLibrary("Features");
		System.loadLibrary("Evaluation");
	}

	public static void main(String argv[]) {
		Features.init_shogun_with_defaults();

		double mul = 2.0;
		DoubleMatrix ground_truth = Load.load_labels("../data/label_train_multiclass.dat");
		DoubleMatrix predicted = Load.load_labels("../data/label_train_multiclass.dat").mmul(mul);

		Labels ground_truth_labels = new Labels(ground_truth);
		Labels predicted_labels = new Labels(predicted);

		MulticlassAccuracy evaluator = new MulticlassAccuracy();
		double accuracy = evaluator.evaluate(predicted_labels, ground_truth_labels);

		System.out.println(accuracy);

		Features.exit_shogun();
	}
}
