#include<stdio.h>
#include <cuda_fp16.h>
//half : one bit for sign, 5 bit for exponent, Fraction/Mantisa: 10 bits
        //single: one bit for sign, 8 bits for exponent, 23 bits for fraction.

__global__ void K1(){

	half a=32.0*31;
	half b=31.0;
	float c=0.0;
	half x=1.0;
	half y=32.0;
	for(int i=0;i<32;i++){
	float d=(float)a*(float) b;
	c=c+d;
	a=__hadd(x,a);//(1,992)
	b=__hadd(y,b);//(32, 31)
	printf("a=%f b=%f c=%f\n",(float)a,(float)b,(float)c);
	}
	printf("a=%f b=%f c=%f\n",(float)a,(float)b,(float)c);
}

main(){
//	half a,b,c;
	c=a+b;
	K1<<<1,1>>>();
	cudaDeviceSynchronize();
}
