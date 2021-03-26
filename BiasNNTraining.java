import java.util.ArrayList;

public class BiasNNTraining
{
    public static void main(String[] args)
    {
        BNN b = getTrained();
        
        
    }
    
    public static BNN getTrained()
    {
        ArrayList<double[]> operations = getOperations();
        
        
        BNN b = train(operations);
        while(b == null)
            b = train(operations);
        
        return b;
    }
    
    public static BNN train(ArrayList<double[]> operations)
    {
        BNN b = new BNN(new int[] {2, 3, 1}, 0.5);
        
        NeuralNetwork.Node[] lastLayer = b.network.get(b.network.size() - 1);
        lastLayer[0].weight[lastLayer[0].weight.length - 1] = 0;
        
        ArrayList<double[]> biases = new ArrayList<>();
        double[] bias = b.getBias();
        for(int i = 0; i < 16; i++)
        {
            for(int j = 0; j < bias.length; j++)
                bias[j] = 2 * (Math.random() - 0.5) * 0.5;
            biases.add(bias);
        }
        
        
        double[] prevError = new double[3];
        for(int epoc = 0; epoc < 50000; epoc++)
        {
            double totalError = 0;
            for(int task = 0; task < 16; task++)
            {
                double[] operation = operations.get(task);
                bias = biases.get(task);
                
                b.setBias(bias);
                
                for(int i = 0; i < 40; i++)
                {
                    double[] input = new double[] {i % 4 / 2, i % 2};
                    
                    if(i < 4)
                        totalError += b.backProp(input, new double[] {operation[i % 4]});
                    else
                        b.backProp(input, new double[] {operation[i % 4]});
                    
                    if(i % 10 == 9)
                        if(epoc < 15000)
                            b.update(0.01);
                        else
                            b.update(0.01);
                }
                
                biases.set(task, b.getBias());
            }
            
            prevError[2] = prevError[1];
            prevError[1] = prevError[0];
            prevError[0] = totalError;
            
            if(epoc > 15000)
            {
                double constant = (prevError[2] - prevError[1])/(prevError[1] - prevError[0]) - 1;
                System.out.println(totalError + " " + constant);
                
                if(totalError > 0.25)
                    return null;
            }
                
        }
        
        double[][] dotmat = new double[16][16];
        for(int i = 0; i < 16; i++)
             for(int j = 0; j < 16; j++)
                 for(int k = 0; k < 3; k++)
                     dotmat[i][j] += biases.get(i)[k] * biases.get(j)[k];
        
        double[] zeroBias = new double[4];
        b.setBias(new double[3]);
        
        for(int i = 0; i < 4; i++)
        {
            double[] input = new double[] {i % 4 / 2, i % 2};
            zeroBias[i] = b.calc(input)[0];
        }
        
        double[] operation = new double[] {0, 1, 1, 0};
        double[] newbias = new double[3];
        for(int j = 0; j < newbias.length; j++)
            newbias[j] = 2 * (Math.random() - 0.5) * 0.5;
        b.setBias(newbias);
        for(int i = 0; i < 100000; i++)
        {
            
            //System.out.println(b.backProp(new double[] {i % 4 / 2, i % 2}, new double[] {operation[i % 4]}));
            if(i % 4 == 3);
                //b.biasUpdate(1000);
        }
        
        show(biases, 40);
        
        return b;
    }


    //assumes the array is 3d
    public static void show(ArrayList<double[]> biases, int imageSize)
    {
        double max = biases.get(0)[0];
        for(double[] bias : biases)
            for(double value : bias)
                if(Math.abs(value) > max)
                    max = Math.abs(value);
        
        //first and second
        double[][] image = new double[imageSize][imageSize];
        for(double[] bias : biases)
        {
            int pos1 = (int) (((bias[0] / max) + 1) / 2 * (imageSize - 1));
            int pos2 = (int) (((bias[1] / max) + 1) / 2 * (imageSize - 1));
            image[pos1][pos2] = 1;
        }
        show(image);
        
      //second and third
        image = new double[imageSize][imageSize];
        for(double[] bias : biases)
        {
            int pos1 = (int) (((bias[1] / max) + 1) / 2 * (imageSize - 1));
            int pos2 = (int) (((bias[2] / max) + 1) / 2 * (imageSize - 1));
            image[pos1][pos2] = 1;
        }
        show(image);
        
      //third and first
        image = new double[imageSize][imageSize];
        for(double[] bias : biases)
        {
            int pos1 = (int) (((bias[2] / max) + 1) / 2 * (imageSize - 1));
            int pos2 = (int) (((bias[0] / max) + 1) / 2 * (imageSize - 1));
            image[pos1][pos2] = 1;
        }
        show(image);          
    }
    
    private static void show(double[][] image)
    {
        for(double[] line : image)
        {
            for(double value : line)
                if(value == 0)
                    System.out.print("  ");
                else
                    System.out.print("@ ");
            System.out.println();
        }
        
        System.out.println("_____________________________");
    }

    public static List<double[]> getOperations()
    {
        ArrayList<double[]> operations = new ArrayList<>();
        
        for(int i = 0; i < 16; i++)
        {
            double[] operation = new double[4];
            operation[0] = i / 8;
            operation[1] = i % 8 / 4;
            operation[2] = i % 8 % 4 / 2;
            operation[3] = i % 8 % 4 % 2 / 1;
            operations.add(operation);
        }
        
        return operations;
    }
}
