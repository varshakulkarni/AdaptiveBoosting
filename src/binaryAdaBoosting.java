import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Scanner;


public class binaryAdaBoosting {

	public static void main(String[] args) throws FileNotFoundException {
		// TODO Auto-generated method stub
		Scanner s=new Scanner(new File("adaboost-5.txt"));
		FileOutputStream fout = new FileOutputStream("binary_5.txt");
        PrintStream ps = new PrintStream(fout);
        System.setOut(ps);
        
		int noOfIterations=s.nextInt();
		int noOfSamples=s.nextInt();
		double epsilon=s.nextDouble();
		float[] x=new float[noOfSamples];
		int[] y=new int[noOfSamples];
		double[] p=new double[noOfSamples];		
		double[] errProb=new double[2];	
		double Et=0;
		double alphat=0;
		double RightQi=0;
		double WrongQi=0;
		double[] Z=new double[2];
		double Zt=0;
		double bound=1;
		double[] fx=new double[noOfSamples];
		double[] H=new double[noOfSamples];
		String Fx=" ";
		double[] ft = new double[noOfSamples];
		for(int i=0;i<noOfSamples;i++){
			x[i]=s.nextFloat();
		}
		for(int i=0;i<noOfSamples;i++){
			y[i]=s.nextInt();
		}
		for(int i=0;i<noOfSamples;i++){
			p[i]=s.nextDouble();
		}
		double[] e=new double[noOfSamples];
		String q="";
		for(int c=1;c<noOfIterations+1;c++){
			System.out.println("Iteration" + c);
			errProb=calculatingThreshold(x,y,p);
			Et=errProb[2];
			
					
			alphat=(0.5)*(Math.log((1-Et)/Et));
			
			  
			double error=0;	
			if(errProb[0] == 1){
                System.out.println("The selected weak classifier : x<"+errProb[1]+" ");
               for(int r=0;r<noOfSamples;r++){
                if(x[r]<errProb[1])
                {
                	H[r]=1;
                }
                else
                {
                	H[r]=-1;
                }
               }
			}
            else if(errProb[0] == 0){
                System.out.println("The selected weak classifier : x>"+errProb[1]+" ");
			 for(int r=0;r<noOfSamples;r++){
	                if(x[r]<errProb[1])
	                {
	                	H[r]=-1;
	                }
	                else
	                {
	                	H[r]=1;
	                }
	               }
				}
			System.out.println("The error of Ht: " +Et);	//2nd one
			System.out.println("The weight of Ht: " +alphat);
			if(c!=1)
                Fx += " + ";
			 
			if(errProb[0] == 0)
			{	Fx += alphat+"* I(x>"+errProb[1]+")";
			}
			else
			{	Fx += alphat+"* I(x<"+errProb[1]+")";
			}

			
			//normalization factor
			RightQi=Math.exp(-alphat);
			WrongQi=Math.exp(alphat);
			Z=normalizationFactor(x,y,p,RightQi,WrongQi,errProb);
			Zt=Z[0];
			
			System.out.println("The probabilities normalization factor Zt: "+Zt);
			
			//new p[i]
			System.out.print("The probabilities after normalization:");
			for(int i=0;i<x.length;i++)
			{
				p[i]=p[i]/Zt;
				System.out.print(p[i]+ "\t");
			}
			System.out.println("");
			//f(x)
			System.out.println("Boosted classifier : "+Fx+" ");
			//Er
			 ft = calculateft(errProb[1],errProb[0],alphat,ft,x,y);
	            double errorCount=0;
	            for(int j=0;j<noOfSamples;j++)
	            {
	                if(ft[j]<0 && y[j]>0)
	                    errorCount++;
	                else if(ft[j]>0 && y[j]<0)
	                    errorCount++;
	            }
			
			System.out.println("The error of the boosted classifier: "+(double)(errorCount/noOfSamples));
			//bound
			bound*=Zt;
			System.out.println("The bound on Et: " +bound);
			System.out.println("\n");
		}
	}
	
	  public static double[] calculateft(double threshold,double weakClassifier,double weight,double[] ft,float[] x,int[] y)
	    {
	        int n = x.length;
	        if(weakClassifier == 0)
	        {
	            int j=0;
	            for(;x[j]<threshold;j++)
	                ft[j] -= weight;
	            for(;j<n;j++)
	                ft[j] += weight;
	        }
	        else if(weakClassifier == 1)
	        {
	            int j=0;
	            for(;x[j]<threshold;j++)
	                ft[j] += weight;
	            for(;j<n;j++)
	                ft[j] -= weight;
	        }
	        return ft;
	    }
	public static double[] normalizationFactor(float[] x,int[] y,double[] p,double RightQi,double WrongQi,double[] errProb){
		double[] Z=new double[2];
		double Zt=0;
		int e1=0;
		
		if(errProb[0]==1)
		{
			for(int i=0;i<x.length;i++)
			{
				if(x[i]<=errProb[1])
				{
					if(y[i]==-1)
					{
						p[i]=p[i]*WrongQi;
						e1++;
					}
					else
					{
						p[i]=p[i]*RightQi;
					}
				}
				else
				{
					if(y[i]==-1)
					{
						p[i]=p[i]*RightQi;
					}
					else
					{
						p[i]=p[i]*WrongQi;
						e1++;
					}	
				}
			}
		}
		else
		{
			for(int i=0;i<x.length;i++)
			{
				if(x[i]<errProb[1])
				{
					if(y[i]==-1)
					{
						p[i]=p[i]*RightQi;
					}
					else
					{
						p[i]=p[i]*WrongQi;
						e1++;
					}
				}
				else
				{
					if(y[i]==-1)
					{
						p[i]=p[i]*WrongQi;
						e1++;
					}
					else
					{
						p[i]=p[i]*RightQi;
					}	
				}
			}
		}
		//System.out.println("e1 is" + e1);
	for(int i=0;i<x.length;i++)
	{
		Zt=Zt+p[i];
	}
	Z[0]=Zt;
	Z[1]=e1;
	return Z;
	//System.out.println(Zt);
	}
	public static double[] calculatingThreshold(float[] x,int[] y,double[] p){
		ArrayList<Double> listOfThresholds=new ArrayList<Double>();
		double[] errProb=new double[4];
		int thresholdIndex=0;
		int errFinal=0;
		double errMini=2;//random value
		double newArray[]=new double[3];
		for (int i=0;i<x.length-1;i++)
		{
			if(i == 0){
				listOfThresholds.add(x[i] - 1.0);
			}
			if(y[i]!=y[i+1])
			{
				listOfThresholds.add((double) ((x[i]+x[i+1])/2.0));
			}
		}
		listOfThresholds.add(x[x.length-1] + 1.0);
		for(int i=0;i<listOfThresholds.size();i++)
		{
			Double threshold=listOfThresholds.get(i);
			errProb=calculateNoOfErrors(x,y,p,threshold);
			if(errProb[1]<errMini)
			{
				errMini=errProb[1];
				thresholdIndex=i;
				errFinal=(int) errProb[0];
			}
					
		}
		newArray[0]=errFinal; //0 or 1
		//System.out.println("Threshold is: "+listOfThresholds.get(thresholdIndex));
		newArray[1]=listOfThresholds.get(thresholdIndex); //threshold value
		newArray[2]=errMini;	//minimum sum of probabilities
		return newArray;
	}
	public static double[] calculateNoOfErrors(float[] x,int[] y,double[] p,double threshold)
	{
		double Err1=0;
		double Err2=0;
		double[] errProb=new double[2];
		
		for(int i=0;i<x.length;i++)
		{
			if(x[i]<threshold)
			{
				if(y[i]==-1)
				{
					Err1=Err1+p[i];
				}
				if(y[i]==1)
				{
					Err2=Err2+p[i];
				}
			}
			if(x[i]>threshold)
			{
				if(y[i]==1)
				{
					Err1=Err1+p[i];
				}
				if(y[i]==-1)
				{
					Err2=Err2+p[i];
				}
			}
		}
//		int leastNoOfErrors=Math.min(countErr1, countErr2);
		if(Err1<Err2)
		{
			errProb[0]=1;
			errProb[1]=Err1;
		}
		else
		{
			errProb[0]=0;
			errProb[1]=Err2;
		}	
		return errProb;
		
	}	
}
