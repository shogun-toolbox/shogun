using System;
using System.IO;

public class Load {
	public static string[] read_file(string file) {
		StreamReader r = null;
		string[] split = null;
		try {
			r = new StreamReader(file);
			string contents = r.ReadToEnd().Trim();
			split = System.Text.RegularExpressions.Regex.Split(contents, "\\s+", System.Text.RegularExpressions.RegexOptions.None);
		}
		catch (Exception e) {
				Console.WriteLine("erro:{0}", e.Message);
		}
		finally {
			r.Close();
		}
		return split;
 }
	public static double[,] load_numbers(string file) {
		string [] split = read_file(file);
		int len = split.Length/2;
		double[,] result = new double[2, len];

		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < len; j++) {
				result[i, j] = Convert.ToDouble(split[i * len + j]);
			}
		}
		return result;
	}
	public static double[] load_labels(string file) {
		string [] split = read_file(file);
		int len = split.Length;
		double[] result = new double[len];

		for (int i = 0; i < len; i++) {
			result[i] = Convert.ToDouble(split[i]);
		}
		return result;
	}
	public static string[] load_dna(string file) {
		string [] result = read_file(file);
		return result;
	}

	public static string[] load_cubes(String file) {
		string [] result = read_file(file);
		return result;
	}
}
