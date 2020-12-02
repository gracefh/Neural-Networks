/*
 * Author: Grace Huang
 * Date Created: 9/6/2019
 * Period 2
 *
 * A simplified Perceptron implementation that can be generalized to any number of layers. The Perceptron is trained
 * on a data set and changes the weights of the network based on the error before and after changing the weights, using
 * back propagation.
 *
 * Methods in this file:
 * Perceptron(String configFile, String trainingFile, String targetsFile)    constructor for the Perceptron class
 * void   set_weights(String weightFile)                                     sets network's weights according to a file
 * void   randomize_weights(double minnum, double maxnum)                    randomizes weights based on a min and max number
 * void   printWeights()                                                     prints network's weights to console
 * void   printOutputs()                                                     prints network's outputs to console
 * void   printOutputs(String file)                                          prints network's outputs to file
 * double calc_activation_value(int n, int i)                                calculates the activation value of a node
 * void   calculate_all_activations()                                        calculates all activation values in network
 * double error_function()                                                   calculates error between expected and real output
 * double calcAllError()                                                     calculates total error from all training cases
 * void   backprop()                                                         changes weights based on back propagation
 * double activation_function(double x)                                      the activation function used by the network
 * double act_derivative(double x)                                           the derivative of the activation function
 * void   train()                                                            trains the network using back propagation
 * static void main(String[] args)                                           creates and trains a network based on
 *                                                                           user-specified files
 */

import java.io.*;
import java.util.*;

public class Perceptron
{

   public static double[][] activations;       // keeps track of the activation values of all nodes in the network
   private static double[] targetValues;       // keeps track of the target values for the output values
   private static int inputs;                  // the number of input nodes in the neural network
   private static int[] hiddenLayers;          // keeps track of how many nodes there are in each hidden layer
   private static int outputs;                 // the number of output nodes in the neural network
   private static int layers;                  // the number of layers in the network
   private double[][][] weights;               // keeps track of the weights of all connections in the network
   private double minweight;                   // the minimum double that weights can be
   private double maxweight;                   // the maximum double that weights can be
   private double[][]theta_sums;               // sum of the products of weights and corresponding connections to a node
   private double[][]psi_vals;                 // defined as product of the difference between target and real value and
                                               // the derivative of the activation function with value theta for a node

   private int timesToTrain;                                   // amount of times to train the network
   private double lambda;                                      // initial lambda coefficient
   private double error_limit;                                 // error threshold to end training
   private double lambdaChange;                                // factor to change lambda by during training
   private int cases;                                          // the number of training cases
   private double[][] trainingoutputs;                         // training outputs
   private double[][] traininginputs;                          // the inputs corresponding to the training outputs

   /*
    * Constructor for the Perceptron Class. Initializes the layers variable as well as the activations and weights arrays.
    *
    * @param configFile: The configuration file for the network that contains the inputs, hidden layer nodes, and outputs
    * @param trainingFile: The training file for the network, which contains the training parameters
    * @param targetsFile: The file which contains the training inputs and outputs
    *
    * @throws FileNotFoundException if any of the 3 files aren't found
    */
   public Perceptron(String configFile, String trainingFile, String targetsFile) throws FileNotFoundException
   {
      Scanner scan = new Scanner(new File(configFile));

      inputs = scan.nextInt();
      hiddenLayers = new int[scan.nextInt()];
      for (int i = 0; i < hiddenLayers.length; i++)
      {
         hiddenLayers[i] = scan.nextInt();
      }
      outputs = scan.nextInt();
      minweight = scan.nextDouble();
      maxweight = scan.nextDouble();

      layers = 1 + 1 + hiddenLayers.length;              // the total number of layers in the neural network

      targetValues = new double[outputs];                // the target output array
      activations = new double[layers][];
      activations[0] = new double[inputs];               // the first column contains all the input nodes
      activations[layers - 1] = new double[outputs];     // the last column contains all the output nodes
      theta_sums = new double[layers][];                 // initialize the theta_sums 2 D array
      theta_sums[0] = new double[inputs];
      theta_sums[layers - 1] = new double[outputs];
      psi_vals = new double[layers][];                   // initialize the psi_vals 2 D array
      psi_vals[0] = new double[inputs];
      psi_vals[layers - 1] = new double[outputs];
      int max_leftmost = inputs;                         // the max index of the left node in all weight relations
      int max_rightmost = outputs;                       // the max index of the right node in all weight relations


      for (int i = 0; i < hiddenLayers.length; i++)
      {
         int numNodes = hiddenLayers[i];
         activations[i + 1] = new double[numNodes];
         theta_sums[i + 1] = new double[numNodes];
         psi_vals[i + 1] = new double[numNodes];

         max_leftmost = Math.max(max_leftmost, numNodes);
         max_rightmost = Math.max(max_rightmost, numNodes);
      }

      for (int i = 0; i < layers; i++) // initialize activations array
      {
         for (int j = 0; j < activations[i].length; j++)
         {
            activations[i][j] = 0.0;
            theta_sums[i][j] = 0.0;
            psi_vals[i][j] = 0.0;
         }
      }

      // The first parameter of the array is layers-1 because there is 1 less layer of connections than layers of activation
      weights = new double[layers - 1][max_leftmost][max_rightmost];

      scan = new Scanner(new File(trainingFile));

      // The following values are all given by the training file.
      timesToTrain = scan.nextInt();
      lambda = scan.nextDouble();
      error_limit = scan.nextDouble();
      lambdaChange = scan.nextDouble();

      scan = new Scanner(new File(targetsFile));

      cases = scan.nextInt();

      traininginputs = new double[cases][inputs];
      trainingoutputs = new double[cases][outputs];

      for (int x = 0; x < cases; x++)
      {
         for (int in = 0; in < inputs; in++)
         {
            traininginputs[x][in] = scan.nextDouble();

         }
      }

      for (int x = 0; x < cases; x++)
      {
         for (int ou = 0; ou < outputs; ou++)
         {
            trainingoutputs[x][ou] = scan.nextDouble();
         }
      }
   }

   /*
    * Sets the weights of the connections in the model based on the weight file given by the user.
    *
    * @param weightFile: The file containing the weights
    *
    * @throws FileNotFoundException if the file can't be found
    */
   public void set_weights(String weightFile) throws FileNotFoundException
   {
      Scanner scan = new Scanner(new File(weightFile));
      for (int lev = 0; lev < weights.length; lev++)
      {
         for (int left = 0; left < weights[lev].length; left++)
         {
            for (int right = 0; right < weights[0][left].length; right++)
            {
               weights[lev][left][right] = scan.nextDouble();
            }
         }
      }
   }

   /*
    * Randomizes the weights of the network
    *
    * @param minnum: the minimum number the weight can be, inclusive
    * @param maxnum: the maximum number the weight can be, inclusive
    */
   public void randomize_weights(double minnum, double maxnum)
   {
      for (int m = 0; m < weights.length; m++)
      {
         for (int j = 0; j < weights[m].length; j++)
         {
            for (int k = 0; k < weights[m][j].length; k++)
            {
               weights[m][j][k] = Math.random() * (maxnum - minnum) + minnum;
            }
         }
      }
   }

   /*
    * Prints the weights of the network to the console
    */
   public void printWeights()
   {
      for (int m = 0; m < weights.length; m++)
      {
         for (int j = 0; j < weights[m].length; j++)
         {
            for (int k = 0; k < weights[m][j].length; k++)
            {
               System.out.print(weights[m][j][k] + " ");
            }
            System.out.println();
         }
      } // for (int m = 0; m < weights.length; m++)
   }

   /*
    * Prints the outputs of the neural network to the console
    */
   public void printOutputs()
   {
      for (int tc = 0; tc < cases; tc++)
      {
         activations[0] = traininginputs[tc];
         targetValues = trainingoutputs[tc];
         calculate_all_activations();

         System.out.println();
         System.out.println("Case " + (tc + 1));
         for (int i = 0; i < trainingoutputs[tc].length; i++)
         {
            System.out.println("Target Output: " + targetValues[i] + " Actual Output: " + activations[layers - 1][i]);
         }
      } // for (int tc = 0; tc < cases; tc++)
   }

   /*
    * Prints the outputs of the neural network to a file
    */
   public void printOutputs(String file)
   {
      try
      {
         BufferedWriter out = new BufferedWriter(new FileWriter(file, false));
         for (int tc = 0; tc < cases; tc++)
         {
            activations[0] = traininginputs[tc];
            targetValues = trainingoutputs[tc];
            calculate_all_activations();

            for (int i = 0; i < trainingoutputs[tc].length; i++)
            {
               double val = activations[layers-1][i];

               out.write(" ");
               out.write(((Double) val).toString());
            }
         } // for (int tc = 0; tc < cases; tc++)
         out.close();
      } // try
      catch (Exception e){
         System.out.println(e);
      }
   }

   /*
    * This function calculates the activation value of a node. The parameters, n and i, determine the node for
    * which the activation value is to be calculated. The raw activation value calculated by multiplying together the
    * activations and weights of the nodes connected to the desired node is scaled using the activation_function to
    * get the final activation value.
    *
    * This function also updates the theta_sums array.
    */
   public double calc_activation_value(int n, int i)
   {
      double act;
      double weight;
      double sum = 0.0;
      double activation_value = 0.0;

      for (int j = 0; j < activations[n - 1].length; j++) // the left node
      {
         act = activations[n - 1][j];
         weight = weights[n - 1][j][i];

         sum += act * weight;
      }
      theta_sums[n][i] = sum;
      activation_value = activation_function(sum);

      return activation_value;
   }

   /*
    * calculates the values of all the activation nodes in the neural network
    */
   private void calculate_all_activations()
   {
      for (int lay = 1; lay < layers; lay++)
      {
         for (int node = 0; node < activations[lay].length; node++)
         {
            activations[lay][node] = calc_activation_value(lay, node);
         }
      }
   }

   /*
    * the error function, based on the target value and the output value of the network
    */
   public double error_function()
   {
      double error = 0.0;
      for (int i = 0; i < outputs; i++)
      {
         double case_err = (targetValues[i] - activations[layers - 1][i]);
         error += case_err * case_err;
      }
      return error / 2.0;
   }

   /*
    * Calculates the total error of the outputs of the network compared to the target outputs
    */
   public double calcAllError()
   {
      double totalError = 0.0;
      for (int trainingcase = 0; trainingcase < trainingoutputs.length; trainingcase++)
      {
         activations[0] = traininginputs[trainingcase];
         targetValues = trainingoutputs[trainingcase];
         calculate_all_activations();
         double cur_err = error_function();
         totalError += cur_err * cur_err;
      }
      return Math.sqrt(totalError);
   }

   /*
    * Uses back propagation to implement changes in weights
    */
   public void backprop()
   {
      int last = layers - 1;
      for(int node = 0; node < activations[last].length; node ++) // last layer has a different equation
      {
         psi_vals[last][node] = (targetValues[node] - activations[last][node])*act_derivative(theta_sums[last][node]);
      }
      for(int lay = layers - 2; lay > 0; lay--)                          // calculates starting from second layer
      {
         for(int node = 0; node < activations[lay].length; node++)       // defines which node we're looking at
         {
            double omega_sum = 0.0;
            for (int right = 0; right < activations[lay + 1].length; right++)
            {
               omega_sum += psi_vals[lay + 1][right] * weights[lay][node][right];
               weights[lay][node][right] += lambda*activations[lay][node]*psi_vals[lay+1][right];
            }
            psi_vals[lay][node] = omega_sum * act_derivative(theta_sums[lay][node]);
         }
      } //  for(int lay = layers - 1; lay > 0; lay--)
      for(int left = 0; left < activations[0].length; left++)
      {
         for(int right = 0; right < activations[1].length; right++)
         {
            weights[0][left][right] += lambda*activations[0][left]*psi_vals[1][right];
         }
      }
   }

   /*
    * The activation function used in the neural network.
    */
   public double activation_function(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   }

   /*
    * The derivative of the activation function
    */
   public double act_derivative(double x)
   {
      double act_val = activation_function(x);
      return act_val * (1.0 - act_val);
   }

   /*
    * Trains the network using back propagation
    */
   public void train()
   {
      int timesRun = 0;
      int testCase = 0;
      double old_error;
      double new_error;

      new_error = 0.0;                                            // initializes error value
      while (timesRun < timesToTrain && calcAllError() > error_limit)
      {
         activations[0] = traininginputs[testCase % cases];       // sets the input layer for the network
         targetValues = trainingoutputs[testCase % cases];        // sets the target outputs for the network
         calculate_all_activations();                             // calculates all the activations
         old_error = error_function();
         backprop();                                              // changes all the weights of the network using
                                                                  // back propagation

         calculate_all_activations();                             // calculates the activations based on the new weights
         new_error = error_function();
         if (new_error > old_error)
         {
            lambda /= lambdaChange;
         }
         else
         {
            lambda *= lambdaChange;    // if the new error doesn't increase, increase the lambda by lambdaChange
         }
         timesRun++;
         testCase++;
      } // while (timesRun < timesToTrain && calcAllError() > error_limit)

      // Prints out  information from the training
      System.out.println("Last Error: " + new_error);
      System.out.println("Last Lambda: " + lambda);
      System.out.println("Times Run: " + timesRun);
      System.out.println("Total Error: " + calcAllError());

      System.out.print("Reason for ending: ");  // prints out the reason for ending training
      if (timesRun == timesToTrain)             // means that the training ended because the number of iterations was reached
      {
         System.out.println("Number of iterations reached " + timesToTrain + " limit");
      }
      else                          // means that the training ended because the error threshold was met
      {
         System.out.println("Total error was less than " + error_limit);
      }
   }


   /*
    * The main function for the Perceptron Class. Creates a Perceptron object, trains the network, and outputs the output
    * values of the network based on the object's initial activations. Note that the training only works if there is
    * one hidden layer because the weight derivative changes depending on the layer.
    */
   public static void main(String[] args)
   {
      try
      {
         Bitmaps bit = new Bitmaps();
         Scanner scan = new Scanner(System.in);
         System.out.println("What is the name of your network configuration file?");
         String config = scan.next();
         System.out.println("What is the name of your training file?");
         String training = scan.next();
         System.out.println("What is the name of your target inputs and outputs file?");
         String targets = scan.next();
         System.out.println("What is the name of your desired outputs file?");
         String outputFile = scan.next();
         System.out.println();
         bit.createfile();

         Perceptron test = new Perceptron(config, training, targets);
         test.randomize_weights(test.minweight, test.maxweight); // set the weights of the network

         test.train();                       // train the networks
         test.printOutputs(outputFile);      // write the outputs of the network to an output file
         test.printOutputs();                // write the outputs of the network to the console as well

         System.out.println();
      }
      catch (Exception e)                    // catches the FileNotFoundException that the constructor might throw
      {
         System.out.println(e);
      }
   }
}
