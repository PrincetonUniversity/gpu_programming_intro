#include <stdio.h>
#include "cumessage.h"

void CPUFunction() {
  printf("Hello world from the CPU.\n");
}

int main() {
  // function to run on the cpu
  CPUFunction();

  // function to run on the gpu
  GPUFunction();

  return 0;
}
