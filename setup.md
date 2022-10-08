# Introduction to GPU Computing

## Setup for live workshop

### Point your browser to `https://bit.ly/36g5YUS`

+ Connect to the eduroam wireless network

+ Open a terminal (e.g., Terminal, PowerShell, PuTTY) [<a href="https://researchcomputing.princeton.edu/education/training/hardware-and-software-requirements-picscie-workshops" target="_blank">click here</a> for help]

+ Please SSH to Adroit in the terminal: `ssh <YourNetID>@adroit.princeton.edu` [click [here](https://researchcomputing.princeton.edu/faq/why-cant-i-login-to-a-clu) for help]

+ If you are new to Linux then consider using the MyAdroit web portal: [https://myadroit.princeton.edu](https://myadroit.princeton.edu)

+ Clone this repo on Adroit:

   ```
   $ cd /scratch/network/$USER
   $ git clone https://github.com/PrincetonUniversity/gpu_programming_intro.git
   ```

+ For the live workshop, to get access to the GPU nodes on Adroit, add this line to your Slurm scripts:

   `#SBATCH --reservation=gpuprimer`

+ Go to the [main page](https://github.com/PrincetonUniversity/gpu_programming_intro) of this repo
