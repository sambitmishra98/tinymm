inline __device__ double make_zero()
{ return 0; }

__global__ void
gimmik_mm_24x8(const double* __restrict__ b, double* __restrict__ c)
{
    const int n = 1000000;
    const int ldb = 1000000;
    const int ldc = 1000000;
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    double bv, csub[6];
    __shared__ double bsub[2][24][32];

    if (i >= n)
      return;

    if (threadIdx.y == 0)
    {
        bsub[0][0][threadIdx.x] = __ldcg(b + i + 0*ldb);
        bsub[0][4][threadIdx.x] = __ldcg(b + i + 4*ldb);
        __barrier_sync(0);
        bv = bsub[0][0][threadIdx.x];
        csub[0] = 1.366025403784439*bv;
        csub[1] = 1.366025403784439*bv;
        csub[2] = -0.3660254037844385*bv;
        csub[3] = -0.3660254037844385*bv;
        csub[4] = 1.366025403784439*bv;
        csub[5] = -0.3660254037844385*bv;
        bv = bsub[0][1][threadIdx.x];
        csub[2] += 1.366025403784439*bv;
        __stcg(c + i + 8*ldc, csub[2]);
        csub[4] += -0.3660254037844385*bv;
        __stcg(c + i + 16*ldc, csub[4]);
        bv = bsub[0][2][threadIdx.x];
        csub[1] += -0.3660254037844385*bv;
        __stcg(c + i + 4*ldc, csub[1]);
        csub[3] += 1.366025403784439*bv;
        __stcg(c + i + 12*ldc, csub[3]);
        bv = bsub[0][3][threadIdx.x];
        bv = bsub[0][4][threadIdx.x];
        csub[0] += -0.3660254037844385*bv;
        __stcg(c + i + 0*ldc, csub[0]);
        csub[5] += 1.366025403784439*bv;
        __stcg(c + i + 20*ldc, csub[5]);
        bv = bsub[0][5][threadIdx.x];
        bv = bsub[0][6][threadIdx.x];
        bv = bsub[0][7][threadIdx.x];
        __barrier_sync(0);
    }
    if (threadIdx.y == 1)
    {
        bsub[0][1][threadIdx.x] = __ldcg(b + i + 1*ldb);
        bsub[0][5][threadIdx.x] = __ldcg(b + i + 5*ldb);
        __barrier_sync(0);
        bv = bsub[0][0][threadIdx.x];
        bv = bsub[0][1][threadIdx.x];
        csub[0] = 1.366025403784439*bv;
        csub[1] = 1.366025403784439*bv;
        csub[3] = -0.3660254037844385*bv;
        csub[5] = -0.3660254037844385*bv;
        bv = bsub[0][2][threadIdx.x];
        csub[2] = -0.3660254037844385*bv;
        csub[4] = 1.366025403784439*bv;
        bv = bsub[0][3][threadIdx.x];
        csub[1] += -0.3660254037844385*bv;
        __stcg(c + i + 5*ldc, csub[1]);
        csub[2] += 1.366025403784439*bv;
        __stcg(c + i + 9*ldc, csub[2]);
        csub[3] += 1.366025403784439*bv;
        __stcg(c + i + 13*ldc, csub[3]);
        csub[4] += -0.3660254037844385*bv;
        __stcg(c + i + 17*ldc, csub[4]);
        bv = bsub[0][4][threadIdx.x];
        bv = bsub[0][5][threadIdx.x];
        csub[0] += -0.3660254037844385*bv;
        __stcg(c + i + 1*ldc, csub[0]);
        csub[5] += 1.366025403784439*bv;
        __stcg(c + i + 21*ldc, csub[5]);
        bv = bsub[0][6][threadIdx.x];
        bv = bsub[0][7][threadIdx.x];
        __barrier_sync(0);
    }
    if (threadIdx.y == 2)
    {
        bsub[0][2][threadIdx.x] = __ldcg(b + i + 2*ldb);
        bsub[0][6][threadIdx.x] = __ldcg(b + i + 6*ldb);
        __barrier_sync(0);
        bv = bsub[0][0][threadIdx.x];
        bv = bsub[0][1][threadIdx.x];
        bv = bsub[0][2][threadIdx.x];
        csub[0] = 1.366025403784439*bv;
        csub[5] = -0.3660254037844385*bv;
        bv = bsub[0][3][threadIdx.x];
        bv = bsub[0][4][threadIdx.x];
        csub[1] = 1.366025403784439*bv;
        csub[2] = -0.3660254037844385*bv;
        csub[3] = -0.3660254037844385*bv;
        csub[4] = 1.366025403784439*bv;
        bv = bsub[0][5][threadIdx.x];
        csub[2] += 1.366025403784439*bv;
        __stcg(c + i + 10*ldc, csub[2]);
        csub[4] += -0.3660254037844385*bv;
        __stcg(c + i + 18*ldc, csub[4]);
        bv = bsub[0][6][threadIdx.x];
        csub[0] += -0.3660254037844385*bv;
        __stcg(c + i + 2*ldc, csub[0]);
        csub[1] += -0.3660254037844385*bv;
        __stcg(c + i + 6*ldc, csub[1]);
        csub[3] += 1.366025403784439*bv;
        __stcg(c + i + 14*ldc, csub[3]);
        csub[5] += 1.366025403784439*bv;
        __stcg(c + i + 22*ldc, csub[5]);
        bv = bsub[0][7][threadIdx.x];
        __barrier_sync(0);
    }
    if (threadIdx.y == 3)
    {
        bsub[0][3][threadIdx.x] = __ldcg(b + i + 3*ldb);
        bsub[0][7][threadIdx.x] = __ldcg(b + i + 7*ldb);
        __barrier_sync(0);
        bv = bsub[0][0][threadIdx.x];
        bv = bsub[0][1][threadIdx.x];
        bv = bsub[0][2][threadIdx.x];
        bv = bsub[0][3][threadIdx.x];
        csub[0] = 1.366025403784439*bv;
        csub[5] = -0.3660254037844385*bv;
        bv = bsub[0][4][threadIdx.x];
        bv = bsub[0][5][threadIdx.x];
        csub[1] = 1.366025403784439*bv;
        csub[3] = -0.3660254037844385*bv;
        bv = bsub[0][6][threadIdx.x];
        csub[2] = -0.3660254037844385*bv;
        csub[4] = 1.366025403784439*bv;
        bv = bsub[0][7][threadIdx.x];
        csub[0] += -0.3660254037844385*bv;
        __stcg(c + i + 3*ldc, csub[0]);
        csub[1] += -0.3660254037844385*bv;
        __stcg(c + i + 7*ldc, csub[1]);
        csub[2] += 1.366025403784439*bv;
        __stcg(c + i + 11*ldc, csub[2]);
        csub[3] += 1.366025403784439*bv;
        __stcg(c + i + 15*ldc, csub[3]);
        csub[4] += -0.3660254037844385*bv;
        __stcg(c + i + 19*ldc, csub[4]);
        csub[5] += 1.366025403784439*bv;
        __stcg(c + i + 23*ldc, csub[5]);
        __barrier_sync(0);
    }
}
