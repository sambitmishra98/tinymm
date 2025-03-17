inline __device__ double make_zero()
{ return 0; }

__global__ void
gimmik_mm_24x24(const double* __restrict__ b, double* __restrict__ c)
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
        bsub[0][8][threadIdx.x] = __ldcg(b + i + 8*ldb);
        bsub[0][12][threadIdx.x] = __ldcg(b + i + 12*ldb);
        bsub[0][16][threadIdx.x] = __ldcg(b + i + 16*ldb);
        bsub[0][20][threadIdx.x] = __ldcg(b + i + 20*ldb);
        __barrier_sync(0);
        bv = bsub[0][0][threadIdx.x];
        csub[4] = -1.366025403784439*bv;
        csub[5] = 0.3660254037844385*bv;
        bv = bsub[0][1][threadIdx.x];
        bv = bsub[0][2][threadIdx.x];
        bv = bsub[0][3][threadIdx.x];
        bv = bsub[0][4][threadIdx.x];
        csub[2] = -1.366025403784439*bv;
        bv = bsub[0][5][threadIdx.x];
        bv = bsub[0][6][threadIdx.x];
        csub[3] = -1.366025403784439*bv;
        bv = bsub[0][7][threadIdx.x];
        bv = bsub[0][8][threadIdx.x];
        csub[0] = -0.3660254037844385*bv;
        bv = bsub[0][9][threadIdx.x];
        bv = bsub[0][10][threadIdx.x];
        csub[1] = -0.3660254037844385*bv;
        bv = bsub[0][11][threadIdx.x];
        bv = bsub[0][12][threadIdx.x];
        csub[2] += -0.3660254037844385*bv;
        __stcg(c + i + 8*ldc, csub[2]);
        bv = bsub[0][13][threadIdx.x];
        bv = bsub[0][14][threadIdx.x];
        csub[3] += -0.3660254037844385*bv;
        __stcg(c + i + 12*ldc, csub[3]);
        bv = bsub[0][15][threadIdx.x];
        bv = bsub[0][16][threadIdx.x];
        csub[0] += -1.366025403784439*bv;
        __stcg(c + i + 0*ldc, csub[0]);
        bv = bsub[0][17][threadIdx.x];
        bv = bsub[0][18][threadIdx.x];
        csub[1] += -1.366025403784439*bv;
        __stcg(c + i + 4*ldc, csub[1]);
        bv = bsub[0][19][threadIdx.x];
        bv = bsub[0][20][threadIdx.x];
        csub[4] += -0.3660254037844385*bv;
        __stcg(c + i + 16*ldc, csub[4]);
        csub[5] += 1.366025403784439*bv;
        __stcg(c + i + 20*ldc, csub[5]);
        bv = bsub[0][21][threadIdx.x];
        bv = bsub[0][22][threadIdx.x];
        bv = bsub[0][23][threadIdx.x];
        __barrier_sync(0);
    }
    if (threadIdx.y == 1)
    {
        bsub[0][1][threadIdx.x] = __ldcg(b + i + 1*ldb);
        bsub[0][5][threadIdx.x] = __ldcg(b + i + 5*ldb);
        bsub[0][9][threadIdx.x] = __ldcg(b + i + 9*ldb);
        bsub[0][13][threadIdx.x] = __ldcg(b + i + 13*ldb);
        bsub[0][17][threadIdx.x] = __ldcg(b + i + 17*ldb);
        bsub[0][21][threadIdx.x] = __ldcg(b + i + 21*ldb);
        __barrier_sync(0);
        bv = bsub[0][0][threadIdx.x];
        bv = bsub[0][1][threadIdx.x];
        csub[4] = -1.366025403784439*bv;
        csub[5] = 0.3660254037844385*bv;
        bv = bsub[0][2][threadIdx.x];
        bv = bsub[0][3][threadIdx.x];
        bv = bsub[0][4][threadIdx.x];
        bv = bsub[0][5][threadIdx.x];
        csub[2] = -1.366025403784439*bv;
        bv = bsub[0][6][threadIdx.x];
        bv = bsub[0][7][threadIdx.x];
        csub[3] = -1.366025403784439*bv;
        bv = bsub[0][8][threadIdx.x];
        csub[0] = 1.366025403784439*bv;
        bv = bsub[0][9][threadIdx.x];
        bv = bsub[0][10][threadIdx.x];
        csub[1] = 1.366025403784439*bv;
        bv = bsub[0][11][threadIdx.x];
        bv = bsub[0][12][threadIdx.x];
        bv = bsub[0][13][threadIdx.x];
        csub[2] += -0.3660254037844385*bv;
        __stcg(c + i + 9*ldc, csub[2]);
        bv = bsub[0][14][threadIdx.x];
        bv = bsub[0][15][threadIdx.x];
        csub[3] += -0.3660254037844385*bv;
        __stcg(c + i + 13*ldc, csub[3]);
        bv = bsub[0][16][threadIdx.x];
        csub[0] += 0.3660254037844385*bv;
        __stcg(c + i + 1*ldc, csub[0]);
        bv = bsub[0][17][threadIdx.x];
        bv = bsub[0][18][threadIdx.x];
        csub[1] += 0.3660254037844385*bv;
        __stcg(c + i + 5*ldc, csub[1]);
        bv = bsub[0][19][threadIdx.x];
        bv = bsub[0][20][threadIdx.x];
        bv = bsub[0][21][threadIdx.x];
        csub[4] += -0.3660254037844385*bv;
        __stcg(c + i + 17*ldc, csub[4]);
        csub[5] += 1.366025403784439*bv;
        __stcg(c + i + 21*ldc, csub[5]);
        bv = bsub[0][22][threadIdx.x];
        bv = bsub[0][23][threadIdx.x];
        __barrier_sync(0);
    }
    if (threadIdx.y == 2)
    {
        bsub[0][2][threadIdx.x] = __ldcg(b + i + 2*ldb);
        bsub[0][6][threadIdx.x] = __ldcg(b + i + 6*ldb);
        bsub[0][10][threadIdx.x] = __ldcg(b + i + 10*ldb);
        bsub[0][14][threadIdx.x] = __ldcg(b + i + 14*ldb);
        bsub[0][18][threadIdx.x] = __ldcg(b + i + 18*ldb);
        bsub[0][22][threadIdx.x] = __ldcg(b + i + 22*ldb);
        __barrier_sync(0);
        bv = bsub[0][0][threadIdx.x];
        bv = bsub[0][1][threadIdx.x];
        bv = bsub[0][2][threadIdx.x];
        csub[4] = -1.366025403784439*bv;
        csub[5] = 0.3660254037844385*bv;
        bv = bsub[0][3][threadIdx.x];
        bv = bsub[0][4][threadIdx.x];
        csub[2] = 0.3660254037844385*bv;
        bv = bsub[0][5][threadIdx.x];
        bv = bsub[0][6][threadIdx.x];
        csub[3] = 0.3660254037844385*bv;
        bv = bsub[0][7][threadIdx.x];
        bv = bsub[0][8][threadIdx.x];
        bv = bsub[0][9][threadIdx.x];
        csub[0] = -0.3660254037844385*bv;
        bv = bsub[0][10][threadIdx.x];
        bv = bsub[0][11][threadIdx.x];
        csub[1] = -0.3660254037844385*bv;
        bv = bsub[0][12][threadIdx.x];
        csub[2] += 1.366025403784439*bv;
        __stcg(c + i + 10*ldc, csub[2]);
        bv = bsub[0][13][threadIdx.x];
        bv = bsub[0][14][threadIdx.x];
        csub[3] += 1.366025403784439*bv;
        __stcg(c + i + 14*ldc, csub[3]);
        bv = bsub[0][15][threadIdx.x];
        bv = bsub[0][16][threadIdx.x];
        bv = bsub[0][17][threadIdx.x];
        csub[0] += -1.366025403784439*bv;
        __stcg(c + i + 2*ldc, csub[0]);
        bv = bsub[0][18][threadIdx.x];
        bv = bsub[0][19][threadIdx.x];
        csub[1] += -1.366025403784439*bv;
        __stcg(c + i + 6*ldc, csub[1]);
        bv = bsub[0][20][threadIdx.x];
        bv = bsub[0][21][threadIdx.x];
        bv = bsub[0][22][threadIdx.x];
        csub[4] += -0.3660254037844385*bv;
        __stcg(c + i + 18*ldc, csub[4]);
        csub[5] += 1.366025403784439*bv;
        __stcg(c + i + 22*ldc, csub[5]);
        bv = bsub[0][23][threadIdx.x];
        __barrier_sync(0);
    }
    if (threadIdx.y == 3)
    {
        bsub[0][3][threadIdx.x] = __ldcg(b + i + 3*ldb);
        bsub[0][7][threadIdx.x] = __ldcg(b + i + 7*ldb);
        bsub[0][11][threadIdx.x] = __ldcg(b + i + 11*ldb);
        bsub[0][15][threadIdx.x] = __ldcg(b + i + 15*ldb);
        bsub[0][19][threadIdx.x] = __ldcg(b + i + 19*ldb);
        bsub[0][23][threadIdx.x] = __ldcg(b + i + 23*ldb);
        __barrier_sync(0);
        bv = bsub[0][0][threadIdx.x];
        bv = bsub[0][1][threadIdx.x];
        bv = bsub[0][2][threadIdx.x];
        bv = bsub[0][3][threadIdx.x];
        csub[4] = -1.366025403784439*bv;
        csub[5] = 0.3660254037844385*bv;
        bv = bsub[0][4][threadIdx.x];
        bv = bsub[0][5][threadIdx.x];
        csub[2] = 0.3660254037844385*bv;
        bv = bsub[0][6][threadIdx.x];
        bv = bsub[0][7][threadIdx.x];
        csub[3] = 0.3660254037844385*bv;
        bv = bsub[0][8][threadIdx.x];
        bv = bsub[0][9][threadIdx.x];
        csub[0] = 1.366025403784439*bv;
        bv = bsub[0][10][threadIdx.x];
        bv = bsub[0][11][threadIdx.x];
        csub[1] = 1.366025403784439*bv;
        bv = bsub[0][12][threadIdx.x];
        bv = bsub[0][13][threadIdx.x];
        csub[2] += 1.366025403784439*bv;
        __stcg(c + i + 11*ldc, csub[2]);
        bv = bsub[0][14][threadIdx.x];
        bv = bsub[0][15][threadIdx.x];
        csub[3] += 1.366025403784439*bv;
        __stcg(c + i + 15*ldc, csub[3]);
        bv = bsub[0][16][threadIdx.x];
        bv = bsub[0][17][threadIdx.x];
        csub[0] += 0.3660254037844385*bv;
        __stcg(c + i + 3*ldc, csub[0]);
        bv = bsub[0][18][threadIdx.x];
        bv = bsub[0][19][threadIdx.x];
        csub[1] += 0.3660254037844385*bv;
        __stcg(c + i + 7*ldc, csub[1]);
        bv = bsub[0][20][threadIdx.x];
        bv = bsub[0][21][threadIdx.x];
        bv = bsub[0][22][threadIdx.x];
        bv = bsub[0][23][threadIdx.x];
        csub[4] += -0.3660254037844385*bv;
        __stcg(c + i + 19*ldc, csub[4]);
        csub[5] += 1.366025403784439*bv;
        __stcg(c + i + 23*ldc, csub[5]);
        __barrier_sync(0);
    }
}
