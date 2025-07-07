# Introduction to GPU Computing

## Setup for live workshop

### Point your browser to `https://bit.ly/36g5YUS`

+ Connect to the eduroam wireless network

+ Open a terminal (e.g., Terminal, PowerShell, PuTTY)

+ Request an [account on Adroit](https://forms.rc.princeton.edu/registration/?q=adroit).

+ Please SSH to Adroit in the terminal: `ssh <YourNetID>@adroit.princeton.edu` [click [here](https://researchcomputing.princeton.edu/faq/why-cant-i-login-to-a-clu) for help]

+ If you are new to Linux then consider using the MyAdroit web portal: [https://myadroit.princeton.edu](https://myadroit.princeton.edu) (VPN required from off-campus)

+ Clone this repo on Adroit:

   ```
   $ cd /scratch/network/$USER
   $ git clone https://github.com/PrincetonUniversity/gpu_programming_intro.git
   $ cd gpu_programming_intro
   ```

+ For the live workshop, to get priority access to the GPU nodes on Adroit, add this line to your Slurm scripts:

   `$ sbatch --reservation=gpuprimer job.slurm`

+ Go to the [main page](https://github.com/PrincetonUniversity/gpu_programming_intro) of this repo
