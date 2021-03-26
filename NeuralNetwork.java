import java.io.*;
import java.util.*;

public class NeuralNetwork
{
    List<Node[]> network = new ArrayList<>();

    Functions f = new ReLUSig();
    
    
    /**
     * Construct Neural Network
     * @param dim   int array with width of each layer
     * @param scale bound of random initialization
     */
    public NeuralNetwork(int[] dim, double scale)
    {
        for(int i = 0; i < dim.length; i++)
        {
            Node[] layer = new Node[dim[i]];//layers are arrays of nodes
            for(int j = 0; j < layer.length; j++)
                layer[j] = new Node((i == 0) ? 0 : dim[i - 1], scale);//initializes weights scale
            network.add(layer);
        }
    }
    
    public void copy(NeuralNetwork nn)
    {
        for(Node[] layer : nn.network)
            network.add(layer);
        f = nn.f;
    }
    
    
    /**Forward passes input through the network
     * 
     * @param input Input vector
     * @return      output vector
     */
    public double[] calc(double[] input)
    {
        for(int i = 0; i < network.get(0).length; i++)//set input layer
            network.get(0)[i].val = input[i];
        
        int layer;
        for(layer = 1; layer < network.size() - 1; layer++)                 //iterate forward through network layers
            for(Node node: network.get(layer))                              //iterate through each node
                node.val = f.activate(node.calc(network.get(layer - 1)));   //set val to activation function applyed to preAct

        double[] output = new double[network.get(layer).length];//initialize output vector

        for(int i = 0; i < network.get(layer).length; i++)
            output[i] = network.get(layer)[i].calc(network.get(layer - 1));//sets output layer preAct
        
        output = f.activateOut(output);
        
        for(int i = 0; i < network.get(layer).length; i++)
            network.get(layer)[i].val = output[i];//sets last layer val to activate out applied to preAct
        return output;
    }
    
    /**Backward passes gradient through the network
     * 
     * @param input     Input vector
     * @param expected  Expected output vector
     * @return          Total error of network
     */

    public double backProp(double[] input, double[] expected)
    {
        double[] output = calc(input);
        int layer = network.size() - 1;
        double[] gradVec;     //saved vector used to calc new gradient
        double[] gradStor = new double[network.get(layer - 1).length];  //accumulated vector used to set to gradvec
                
        int counter = 0;
        for(Node node: network.get(layer))
            add(gradStor, node.grad(f.derOut(node.preAct, node.val, expected[counter++]), network.get(layer - 1)));//initialize output gradient
        
        for(layer = layer - 1; layer > 0; layer--)
        {
            gradVec = gradStor;
            gradStor = new double[network.get(layer - 1).length];
            counter = 0;
            for(Node node: network.get(layer))
                add(gradStor, node.grad(gradVec[counter++] * f.der(node.preAct, node.val), network.get(layer - 1)));//compute node gradient and accumulate gradStor
        }
        
        return f.error(output, expected);
    }
    
    private void add(double[] base, double[] input)// short way to add in place
    {
        for(int i = 0; i < base.length; i++)
            base[i] += input[i];
    }
    
    public void update(double rate)
    {
        for(int layer = 1; layer < network.size(); layer++)
            for(Node node: network.get(layer))
            {
                for(int i = 0; i < node.weight.length; i++)//update weights by negative gradient
                    node.weight[i] -= node.grad[i] * rate;
                node.grad = null;
            }
    }
    
    
    public void clear()
    {
        for(int i = 1; i < network.size(); i++)
            for(Node node : network.get(i))
                    node.grad = null;
    }
    
    public class Node
    {
        double val;
        double preAct;
        double[] weight = null;
        double[] grad = null;
        
        
        public Node(int nodes, double scale)
        {
            if(nodes == 0)
                return;
            weight = new double[nodes + 1];
            for(int i = 0; i < weight.length; i++)
                weight[i] = 2 * (Math.random() - 0.5) * scale;
        }

        public double calc(Node[] nodes)
        {
            preAct = weight[weight.length - 1];
            for(int i = 0; i < nodes.length; i++)
                preAct += weight[i] * nodes[i].val;

            return preAct;
        }

        public double[] grad(double error, Node[] prev)
        {
            if(grad == null)
                grad = new double[weight.length];
            double[] nodeGrad = new double[prev.length];
            for(int i = 0; i < weight.length - 1; i++)
            {
                grad[i] += prev[i].val * error;
                nodeGrad[i] = error * weight[i];
            }
            grad[weight.length - 1] += error;
            return nodeGrad;
        }
    }
    
    
    
    public NeuralNetwork(String fileName)
    {
        File file = new File(fileName);
        try
        {
            Scanner scanner = new Scanner(file);
            
            String[] inString = scanner.nextLine().replace("[", "").replace("]", "").split(", ");
            int[] dim = new int[inString.length];
            
            for(int i = 0; i < inString.length; i++)
                dim[i] = Integer.parseInt(inString[i]);
            
            int nodes = 0;
            for(int value: dim)
            {
                Node[] layer = new Node[value];
                for(int i = 0; i < value; i++)
                    layer[i] = new Node(nodes, 0);
                network.add(layer);
                nodes = value;
            }
            
            for(int layer = 1; layer < network.size(); layer++)
                for(Node node: network.get(layer))
                {
                    inString = scanner.nextLine().replace("[", "").replace("]", "").split(", ");
                    for(int i = 0; i < node.weight.length; i++)
                        node.weight[i] = Double.parseDouble(inString[i]);
                }
        }
        catch (FileNotFoundException e)
        {
            System.out.print("File could not be found");
            return;
        } 
        catch (Exception e)
        {
            System.out.print("File data error");
        }
    }
    
    public void save(String fileName)
    {
        int[] dim = new int[network.size()];
        Arrays.parallelSetAll(dim, x -> network.get(x).length);
        
        try
        {
            File file = new File(fileName);
            file.createNewFile();
            
            FileWriter fileWriter = new FileWriter(file.getAbsolutePath());
            
            fileWriter.write(Arrays.toString(dim) + "\n");
            
            for(int layer = 1; layer < network.size(); layer++)
                for(Node node: network.get(layer))
                    fileWriter.write(Arrays.toString(node.weight) + "\n");
            
            fileWriter.flush();
            fileWriter.close();
        }
        catch (Exception e)
        {
            System.out.print("network could not be saved");
        }
    }
    
    public interface Functions
    {
        double activate(double input);
        double der(double input, double output);
        double[] activateOut(double[] input);
        double derOut(double preAct, double val, double exp);
        double error(double[] output, double[] expected);
    }
    
    public static class ReLUSig implements Functions
    {
        public double activate(double input)
        {
            return Math.max(input, 0);
        }
        public double der(double input, double output)
        {
            return Math.max(input / Math.abs(input), 0);
        }
        public double[] activateOut(double[] input)
        {
            double[] ret = new double[input.length];
            for(int i = 0; i < ret.length; i++)
                ret[i] = 1 / (1 + Math.exp(-input[i]));
            return ret;
        }
        public double derOut(double preAct, double val, double exp)
        {
            return (val - exp) * val * (1 - val);
        }
        public double error(double[] output, double[] expected)
        {
            double error = 0;
            for(int i = 0; i < output.length; i++)
                error += 0.5 * Math.pow(output[i] - expected[i], 2);
            return error;
        }
        
    }

}
