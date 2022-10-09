#include <stdio.h>

void printLoopIndex() {
  int N = 100;
  for (int i = 0; i < N; ++i)
    printf("%d\n", i);
}

int main() {
  // function to run on the cpu
  printLoopIndex();
}
