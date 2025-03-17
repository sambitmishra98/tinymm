inline __device__ double make_zero()
{ return 0; }

__global__ void
gimmik_mm_24x24(const double* __restrict__ b, double* __restrict__ c)
{
    const int n = 1000000;
    const int ldb = 1000000;
    const int ldc = 1000000;
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    double dotp;

    if (i < n)
    {
        dotp = (-0.3660254037844385*b[i + 8*ldb] + -1.366025403784439*b[i + 16*ldb]);
        c[i + 0*ldc] = dotp;
        dotp = (1.366025403784439*b[i + 8*ldb] + 0.3660254037844385*b[i + 16*ldb]);
        c[i + 1*ldc] = dotp;
        dotp = (-0.3660254037844385*b[i + 9*ldb] + -1.366025403784439*b[i + 17*ldb]);
        c[i + 2*ldc] = dotp;
        dotp = (1.366025403784439*b[i + 9*ldb] + 0.3660254037844385*b[i + 17*ldb]);
        c[i + 3*ldc] = dotp;
        dotp = (-0.3660254037844385*b[i + 10*ldb] + -1.366025403784439*b[i + 18*ldb]);
        c[i + 4*ldc] = dotp;
        dotp = (1.366025403784439*b[i + 10*ldb] + 0.3660254037844385*b[i + 18*ldb]);
        c[i + 5*ldc] = dotp;
        dotp = (-0.3660254037844385*b[i + 11*ldb] + -1.366025403784439*b[i + 19*ldb]);
        c[i + 6*ldc] = dotp;
        dotp = (1.366025403784439*b[i + 11*ldb] + 0.3660254037844385*b[i + 19*ldb]);
        c[i + 7*ldc] = dotp;
        dotp = (-1.366025403784439*b[i + 4*ldb] + -0.3660254037844385*b[i + 12*ldb]);
        c[i + 8*ldc] = dotp;
        dotp = (-1.366025403784439*b[i + 5*ldb] + -0.3660254037844385*b[i + 13*ldb]);
        c[i + 9*ldc] = dotp;
        dotp = (0.3660254037844385*b[i + 4*ldb] + 1.366025403784439*b[i + 12*ldb]);
        c[i + 10*ldc] = dotp;
        dotp = (0.3660254037844385*b[i + 5*ldb] + 1.366025403784439*b[i + 13*ldb]);
        c[i + 11*ldc] = dotp;
        dotp = (-1.366025403784439*b[i + 6*ldb] + -0.3660254037844385*b[i + 14*ldb]);
        c[i + 12*ldc] = dotp;
        dotp = (-1.366025403784439*b[i + 7*ldb] + -0.3660254037844385*b[i + 15*ldb]);
        c[i + 13*ldc] = dotp;
        dotp = (0.3660254037844385*b[i + 6*ldb] + 1.366025403784439*b[i + 14*ldb]);
        c[i + 14*ldc] = dotp;
        dotp = (0.3660254037844385*b[i + 7*ldb] + 1.366025403784439*b[i + 15*ldb]);
        c[i + 15*ldc] = dotp;
        dotp = (-1.366025403784439*b[i + 0*ldb] + -0.3660254037844385*b[i + 20*ldb]);
        c[i + 16*ldc] = dotp;
        dotp = (-1.366025403784439*b[i + 1*ldb] + -0.3660254037844385*b[i + 21*ldb]);
        c[i + 17*ldc] = dotp;
        dotp = (-1.366025403784439*b[i + 2*ldb] + -0.3660254037844385*b[i + 22*ldb]);
        c[i + 18*ldc] = dotp;
        dotp = (-1.366025403784439*b[i + 3*ldb] + -0.3660254037844385*b[i + 23*ldb]);
        c[i + 19*ldc] = dotp;
        dotp = (0.3660254037844385*b[i + 0*ldb] + 1.366025403784439*b[i + 20*ldb]);
        c[i + 20*ldc] = dotp;
        dotp = (0.3660254037844385*b[i + 1*ldb] + 1.366025403784439*b[i + 21*ldb]);
        c[i + 21*ldc] = dotp;
        dotp = (0.3660254037844385*b[i + 2*ldb] + 1.366025403784439*b[i + 22*ldb]);
        c[i + 22*ldc] = dotp;
        dotp = (0.3660254037844385*b[i + 3*ldb] + 1.366025403784439*b[i + 23*ldb]);
        c[i + 23*ldc] = dotp;
    }
}
