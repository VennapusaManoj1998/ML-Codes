import javax.swing.*;
import java.awt.*;
import java.util.Random;

public class PerceptronVisualization extends JFrame {
    private static final int WIDTH = 1000;
    private static final int HEIGHT = 1000;
    private static final int POINT_SIZE = 10;

    private double[] weights;
    private int[] inputs;
    private int[] labels;
    private int numPoints;
    private int numMisclassified;

    public PerceptronVisualization() {
        setTitle("Perceptron Visualization");
        setSize(WIDTH, HEIGHT);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);

        weights = new double[3];
        inputs = new int[3];
        labels = new int[20];
        numPoints = 20;
        numMisclassified = numPoints;

        generateTrainingData();
        initializeWeights();

        JPanel panel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                Graphics2D g2d = (Graphics2D) g;
                drawLine(g2d);
                drawPoints(g2d);
            }
        };
        add(panel);
        setVisible(true);

        trainPerceptron(panel);
    }

    private void generateTrainingData() {
        Random random = new Random();
        int a = random.nextInt(HEIGHT);
        int b = random.nextInt(HEIGHT);
        int c = random.nextInt(HEIGHT);

        for (int i = 0; i < numPoints; i++) {
            int x = random.nextInt(WIDTH);
            int y = random.nextInt(HEIGHT);
            int label = (a * x + b * y + c >= 0) ? 1 : -1;

            inputs[i * 3] = x;
            inputs[i * 3 + 1] = y;
            inputs[i * 3 + 2] = 1;
            labels[i] = label;
        }
    }

    private void initializeWeights() {
        Random random = new Random();

        for (int i = 0; i < weights.length; i++) {
            weights[i] = random.nextDouble();
        }
    }

    private void drawLine(Graphics2D g2d) {
        g2d.setColor(Color.GREEN);

        int x1 = 0;
        int y1 = (int) (-weights[2] - weights[0] * x1) / (int) weights[1];
        int x2 = WIDTH;
        int y2 = (int) (-weights[2] - weights[0] * x2) / (int) weights[1];

        g2d.drawLine(x1, y1, x2, y2);
    }

    private void drawPoints(Graphics2D g2d) {
        for (int i = 0; i < numPoints; i++) {
            int x = inputs[i * 3];
            int y = inputs[i * 3 + 1];
            int label = labels[i];

            g2d.setColor(label == 1 ? Color.BLUE : Color.RED);
            g2d.fillOval(x - POINT_SIZE / 2, y - POINT_SIZE / 2, POINT_SIZE, POINT_SIZE);
        }
    }

    private void trainPerceptron(JPanel panel) {
        double learningRate = 0.0001;
        int epoch = 0;

        while (numMisclassified > 0) {
            numMisclassified = 0;

            for (int i = 0; i < numPoints; i++) {
                int label = labels[i];
                int output = classify(inputs[i * 3], inputs[i * 3 + 1]);
                                // Calculate the dot product of weights and inputs
                                double dotProduct = 0;
                                for (int j = 0; j < weights.length; j++) {
                                    dotProduct += weights[j] * inputs[i * 3 + j];
                                }
                
                                // Apply the activation function to get the predicted label
                                int predictedLabel = (dotProduct >= 0) ? 1 : -1;
                
                                // Update the weights if the predicted label is incorrect
                                if (predictedLabel != label) {
                                    numMisclassified++;
                
                                    for (int j = 0; j < weights.length; j++) {
                                        weights[j] += learningRate * (label - predictedLabel) * inputs[i * 3 + j];
                                    }
                                }
                            }
                
                            epoch++;
                            System.out.println("Epoch: " + epoch + ", Misclassified: " + numMisclassified);
                
                            // Update the GUI to visualize the current line
                            panel.repaint();
                
                            // Pause for a short time to visualize the animation
                            try {
                                Thread.sleep(100);
                            } catch (InterruptedException e) {
                                e.printStackTrace();
                            }
                        }
                
                        System.out.println("Training completed!");
                    }
                
                    private int classify(int x, int y) {
                        double sum = weights[0] * x + weights[1] * y + weights[2];
                        return (sum >= 0) ? 1 : -1;
                    }
                
                    public static void main(String[] args) {
                        new PerceptronVisualization();
                    }
                }
                

