import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Scanner;


public class realAdaBoosting {

	public static void main(String[] args) throws FileNotFoundException {
		Scanner s=new Scanner(new File("adaboost-5.txt"));
		FileOutputStream fout = new FileOutputStream("real_5.txt");
        PrintStream ps = new PrintStream(fout);
        System.setOut(ps);
        
		int noOfIterations=s.nextInt();
		int noOfSamples=s.nextInt();
		double epsilon=s.nextDouble();
		float[] x=new float[noOfSamples];
		int[] y=new int[noOfSamples];
		double[] p=new double[noOfSamples];
		double[] errProb=new double[4];	
		double[] correctWrongProbs=new double[4]; //to check if the prob. is correctly classified or wrong
		double G=0;
		double CtPos=0,CtNeg=0;
		double Zt=0;
		double[] Z=new double[4]; // to store temporary values of prob and Z
		double []Ft=new double[x.length];
		double Bound=1;
		int[] e=new int[noOfSamples];
		
		for(int i=0;i<noOfSamples;i++){
			x[i]=s.nextFloat();
		}
		for(int i=0;i<noOfSamples;i++){
			y[i]=s.nextInt();
		}
		for(int i=0;i<noOfSamples;i++){
			p[i]=s.nextDouble();
		}
		for(int c=1;c<=noOfIterations;c++){
			float error=0;
			System.out.println("Iteration " +c);
		//calculating threshold
		errProb=calculatingThreshold(x,y,p);
		if(errProb[0] == 1)
            System.out.println("The selected weak classifier Ht: x<"+errProb[1]+" ");
        else if(errProb[0] == 0)
            System.out.println("The selected weak classifier Ht: x>"+errProb[1]+" ");
		
		//calculating G
		correctWrongProbs=correctWrongClassifyingFunction(x,y,p,errProb);
		G=Math.sqrt(correctWrongProbs[0]*correctWrongProbs[3])+Math.sqrt(correctWrongProbs[1]*correctWrongProbs[2]);
		System.out.println("The G error value of Ht:" +G);
		
		//calculating weight CtPos and CtNeg
		CtPos=(0.50)*(Math.log((correctWrongProbs[0]+epsilon)/(correctWrongProbs[3]+epsilon)));
		CtNeg=(0.50)*(Math.log((correctWrongProbs[2]+epsilon)/(correctWrongProbs[1]+epsilon)));
		System.out.println("The weights Ct+, Ct-:" +CtPos+" ," +CtNeg);
		
		
		//calculating normalized probabilities and normalization factor
		Zt=normalizationFactor(x,y,p,errProb,CtPos,CtNeg);
		System.out.println("The probabilities normalization factor Zt :" +Zt);
		System.out.print("Probabilities after normalization:");
		//Probabilities after normalization
		for(int i=0;i<x.length;i++)
		{
			p[i]=p[i]/Zt;
			System.out.print(" " +p[i]);
		}
		System.out.println("");
		//calculating Ft
		Ft=calculatingft(x,y,p,errProb,CtPos,CtNeg,Ft);
		System.out.print("The values ft(xi) for each one of the examples:  ");
		for(int j=0;j<x.length;j++)
        {
            if(j!=0)
                System.out.print(" , ");
            System.out.print(Ft[j]);
            if(Ft[j]>0)
            	e[j]=1;
            else
            	e[j]=-1;
            if(e[j]!=y[j])
            	error++;
        }
		
		System.out.println("");
		System.out.println("The error of the boosted classifier Et: "+(error/noOfSamples));
		Bound*= Zt;
		System.out.println("The bound on Et: "+Bound);
		System.out.println(" ");
		}
	}
	public static double[] calculatingft(float[] x,int[] y,double[] p,double[] errProb,double CtPos,double CtNeg, double[] ft)
	{
	
	
		if (errProb[0]==1)
		{
			for(int j=0;j<x.length;j++)
			{
					if(x[j]<errProb[1])
					{
						ft[j]=ft[j]+CtPos;
					} 
					else
					{
						ft[j]=ft[j]+CtNeg;
					}
			}
		}
		else
		{
			for(int j=0;j<x.length;j++)
			{
					if(x[j]<errProb[1])
					{
						ft[j]=ft[j]+CtNeg;
					} 
					else
					{
						ft[j]=ft[j]+CtPos;
					}
			}
		}
		return ft;
	}
	
	public static double normalizationFactor(float[] x,int[] y,double[] p,double[] errProb,double CtPos,double CtNeg)
	{
	double Z=0;
	if(errProb[0]==1)
	{
		for(int i=0; i<x.length;i++)
		{
			if(errProb[1]>x[i])
			{
				p[i]=p[i]*(Math.exp(-(y[i]*CtPos)));
			}
			else
			{
				p[i]=p[i]*(Math.exp(-(y[i]*CtNeg)));
			}
		}
	}
	else
	{
		for(int i=0; i<x.length;i++)
		{
			if(errProb[1]>x[i])
			{
				p[i]=p[i]*(Math.exp(-(y[i]*CtNeg)));
			}
			else
			{
				p[i]=p[i]*(Math.exp(-(y[i]*CtPos)));
			}
		}
	}
	for(int i=0;i<x.length;i++)
	{
		Z=Z+p[i];
	}
	return Z;
	}
	
	public static double[] correctWrongClassifyingFunction(float[] x,int[] y,double[] p,double[] errProb)
	{
		double PwNeg=0,PwPos=0,PrPos=0,PrNeg=0;
		double[] probsreturn=new double[4];
		if(errProb[0]==1)
		{
			for(int i=0;i<x.length;i++)
			{
				if(x[i]<errProb[1])
				{
					if(y[i]==-1)
					{
						PwNeg+=p[i];
					}
					else
					{
						PrPos+=p[i];
					}
				}
				else
				{
					if(y[i]==-1)
					{
						PrNeg+=p[i];
					}
					else
					{
						PwPos+=p[i];
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
						PrNeg+=p[i];
					}
					else
					{
						PwPos+=p[i];
					}
				}
				else
				{
					if(y[i]==-1)
					{
						PwNeg+=p[i];
					}
					else
					{
						PrPos+=p[i];
					}	
				}
			}
		}
		probsreturn[0]=PrPos;
		probsreturn[1]=PrNeg;
		probsreturn[2]=PwPos;
		probsreturn[3]=PwNeg;
		return probsreturn;
		
	}
	public static double[] calculatingThreshold(float[] x,int[] y,double[] p){
		ArrayList<Double> listOfThresholds=new ArrayList<Double>();
		double[] errProb=new double[4];
		int thresholdIndex=0;
		int errFinal=0;
		double errProbFinal=1000;//random value
		double newArray[]=new double[4];
		double G=0;
		double gFinal=1000;
		for (int i=0;i<x.length-1;i++)
		{
			if(i == 0){
				listOfThresholds.add(x[i] - 0.1);
			}
			if(y[i]!=y[i+1])
			{
				listOfThresholds.add((double) ((x[i]+x[i+1])/2.0));
			}
		}
		listOfThresholds.add(x[x.length-1] + 0.1);
		for(int i=0;i<listOfThresholds.size();i++)
		{
			Double threshold=listOfThresholds.get(i);
			errProb=calculateNoOfErrors(x,y,p,threshold);
			if(errProb[2]<gFinal)
			{
				errProbFinal=errProb[1];
				thresholdIndex=i;
				errFinal=(int) errProb[0];
				gFinal=errProb[2];
			}	
		}
		
		
		newArray[0]=errFinal; //0 or 1
		//System.out.println("Threshold is: "+listOfThresholds.get(thresholdIndex));
		newArray[1]=listOfThresholds.get(thresholdIndex); //threshold value
		newArray[2]=errProbFinal;	//minimum sum of probabilities 
		newArray[3]=gFinal; //min value of G
		//System.out.println("The G error value of Ht:" +gFinal);
		return newArray;
	}
	public static double[] calculateNoOfErrors(float[] x,int[] y,double[] p,double threshold)
	{
		double Err1=0;
		double Err2=0;
		double[] errProb=new double[3];
		double[] probabilities1 = new double[4];
		double[] probabilities2 = new double[4];
		double G1=0,G2=0;
		for(int i=0;i<x.length;i++)
		{
			if(x[i]<threshold)
			{
				if(y[i]==-1)
				{
					Err1=Err1+p[i];
					probabilities1[1]=probabilities1[1]+p[i];
					probabilities2[3]=probabilities2[3]+p[i];
				}
				if(y[i]==1)
				{
					Err2=Err2+p[i];
					probabilities1[2]=probabilities1[2]+p[i];
					probabilities2[0]=probabilities2[0]+p[i];
				}
			}
			if(x[i]>threshold)
			{
				if(y[i]==1)
				{
					Err1=Err1+p[i];
					probabilities1[0]=probabilities1[0]+p[i];
					probabilities2[2]=probabilities2[2]+p[i];
				}
				if(y[i]==-1)
				{
					Err2=Err2+p[i];
					probabilities1[3]=probabilities1[3]+p[i];
					probabilities2[1]=probabilities2[1]+p[i];
				}
			}
		}
		
		G1=Math.sqrt(probabilities1[0]*probabilities1[3])+Math.sqrt(probabilities1[1]*probabilities1[2]);
		G2=Math.sqrt(probabilities2[0]*probabilities2[3])+Math.sqrt(probabilities2[1]*probabilities2[2]);
		if(G1<=G2)
		{
			errProb[0]=1;
			errProb[1]=Err1;
			errProb[2]=G1;
		}
		else
		{
			errProb[0]=0;
			errProb[1]=Err2;
			errProb[2]=G2;
		}	
		return errProb;
		
	}	
}