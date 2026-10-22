#include<stdio.h>
#include<iostream>
#include <cuda_fp16.h>
//half : one bit for sign, 5 bit for exponent, Fraction/Mantisa: 10 bits
        //single: one bit for sign, 8 bits for exponent, 23 bits for fraction.

__global__ void K1(){

	__half a=32.0*31;
	__half b=31.0;
	__half c=0.0;
	__half x=1.0;
	__half y=32.0;
	for(int i=0;i<32;i++){
	 __half d=a* b;
	c=c+d;
	a=x+a;//(1,992)
	b=y+b;//(32, 31)
	printf("a=%f b=%f c=%f\n",(float)a,(float)b,(float)c);
	}
	printf("a=%f b=%f c=%f\n",(float)a,(float)b,(float)c);
}

main(){
	__half a=__float2half(10.5f);
	__half b=__float2half(10.5f);
	__half c=__float2half(10.5f);
	c=static_cast<float>(a)+static_cast<float>(b);
//c= a+b;
    std::cout << "Value: " << __half2float(c) << std::endl;

	K1<<<1,1>>>();
	cudaDeviceSynchronize();
}
