/* Synchronized numerical baseline: the same shared bridge/kernels used by Lisp. */
#include "benchmark_api.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void die(const char *message) { fprintf(stderr,"%s: %s\n",message,tb_last_error()); exit(1); }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static void step(tb_context *context, tb_tensor *a) {
 tb_tensor *b=tb_matmul(context,a,a);
 if(!b || tb_tensor_eval(b)) die("matmul failed");
 tb_tensor_free(b);
}
int main(int argc, char **argv) {
 if(argc!=5 || (strcmp(argv[1],"cpu") && strcmp(argv[1],"gpu"))) {
   fprintf(stderr,"usage: tb_native_benchmark cpu|gpu dimension iterations repeats\n"); return 2;
 }
 int n=atoi(argv[2]), iterations=atoi(argv[3]), repeats=atoi(argv[4]);
 if(n<1 || n>4096 || iterations<1 || iterations>100000 || repeats<1 || repeats>100) return 2;
 tb_context *context=tb_context_new(!strcmp(argv[1],"gpu"));
 if(!context) die("device creation failed");
 size_t size=(size_t)n*n; float *data=malloc(size*sizeof(float));
 if(!data) die("allocation failed");
 for(size_t i=0;i<size;i++) data[i]=1.0f/n;
 int shape[]={n,n}; tb_tensor *a=tb_tensor_float(data,shape,2);
 if(!a) die("tensor creation failed");
 for(int i=0;i<10;i++) step(context,a);
 printf("{\"frontend\":\"c\",\"device\":\"%s\",\"dimension\":%d,\"iterations\":%d,\"seconds\":[",argv[1],n,iterations);
 for(int r=0;r<repeats;r++) {
   double start=now();
   for(int i=0;i<iterations;i++) step(context,a);
   printf("%s%.12g",r ? "," : "",(now()-start)/iterations);
 }
 tb_tensor *check=tb_matmul(context,a,a);
 if(!check || tb_tensor_copy_float(context,check,data,size)) die("result copy failed");
 for(size_t i=0;i<size;i++) if(!isfinite(data[i]) || fabsf(data[i]-1.0f/n)>1e-6f) die("incorrect result");
 printf("],\"verified_value\":%.9g}\n",data[0]);
 tb_tensor_free(check); tb_tensor_free(a); tb_context_free(context); free(data);
 return 0;
}
