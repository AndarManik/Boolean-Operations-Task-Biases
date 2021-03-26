

public class BNN extends NeuralNetwork
{
    public BNN(int[] dim, double scale)
    {
        super(dim, scale);
    }
    
    public double[] getBias()
    {
        int biasSize = 0;
        for(int i = 1; i < network.size() - 1; i++)
            biasSize += network.get(i).length;
        
        double[] bias = new double[biasSize];
        int biasIndex = 0;
        for(int i = 1; i < network.size() - 1; i++)
            for(Node n : network.get(i))
                bias[biasIndex++] = n.weight[n.weight.length - 1];
        
        return bias;
    }
    
    public void setBias(double[] bias)
    {
        int biasIndex = 0;
        for(int i = 1; i < network.size() - 1; i++)
            for(Node n : network.get(i))
                n.weight[n.weight.length - 1] = bias[biasIndex++];
    }
    
    public void biasUpdate(double rate)
    {
        for(int layer = 1; layer < network.size() - 1; layer++)
            for(Node node: network.get(layer))
            {
                node.weight[node.weight.length - 1] -= node.grad[node.weight.length - 1] * rate;
                node.grad = null;
            }
    }
}
