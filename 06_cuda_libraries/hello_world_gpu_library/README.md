# Building a Simple GPU Library

In this exercise we will construct a "hello world" GPU library called `cumessage` and then link and run a code against it.

### Create the Library

Inspect the files that compose the GPU library (`cumessage.h` and `cumessage.cu`) and then look at our main routine in `myapp.cu`:

```bash
$ cd 06_cuda_libraries/hello_world_gpu_library
$ cat cumessage.h
$ cat cumessage.cu
$ cat myapp.cu
```

Note that `myapp.cu` only needs to know about the actual GPU function `GPUfunction` takes no arguments and, of course, returns no arguments. Nothing is said about how that function is implemented. That is done entirelly in the library.

```bash
$ module load cudatoolkit/11.4
$ nvcc -Xcompiler -fPIC -o libcumessage.so -shared cumessage.cu
$ ls -l
```

The last line above is using the CUDA compiler `nvcc` to create a shared library. Add the option "-v" to see more details.
