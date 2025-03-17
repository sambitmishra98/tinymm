inline __device__ double make_zero()
{ return 0; }

__global__ void
gimmik_mm_24x24(const double* __restrict__ b, double* __restrict__ c)
{
    const int n = 1000000;
    const int ldb = 1000000;
    const int ldc = 1000000;
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i < n)
    {
        double bv, csub[24];

        bv = __ldcg(b + i + 0*ldb);
        csub[16] = -1.366025403784439*bv;
        csub[20] = 0.3660254037844385*bv;
        bv = __ldcg(b + i + 1*ldb);
        csub[17] = -1.366025403784439*bv;
        csub[21] = 0.3660254037844385*bv;
        bv = __ldcg(b + i + 2*ldb);
        csub[18] = -1.366025403784439*bv;
        csub[22] = 0.3660254037844385*bv;
        bv = __ldcg(b + i + 3*ldb);
        csub[19] = -1.366025403784439*bv;
        csub[23] = 0.3660254037844385*bv;
        bv = __ldcg(b + i + 4*ldb);
        csub[8] = -1.366025403784439*bv;
        csub[10] = 0.3660254037844385*bv;
        bv = __ldcg(b + i + 5*ldb);
        csub[9] = -1.366025403784439*bv;
        csub[11] = 0.3660254037844385*bv;
        bv = __ldcg(b + i + 6*ldb);
        csub[12] = -1.366025403784439*bv;
        csub[14] = 0.3660254037844385*bv;
        bv = __ldcg(b + i + 7*ldb);
        csub[13] = -1.366025403784439*bv;
        csub[15] = 0.3660254037844385*bv;
        bv = __ldcg(b + i + 8*ldb);
        csub[0] = -0.3660254037844385*bv;
        csub[1] = 1.366025403784439*bv;
        bv = __ldcg(b + i + 9*ldb);
        csub[2] = -0.3660254037844385*bv;
        csub[3] = 1.366025403784439*bv;
        bv = __ldcg(b + i + 10*ldb);
        csub[4] = -0.3660254037844385*bv;
        csub[5] = 1.366025403784439*bv;
        bv = __ldcg(b + i + 11*ldb);
        csub[6] = -0.3660254037844385*bv;
        csub[7] = 1.366025403784439*bv;
        bv = __ldcg(b + i + 12*ldb);
        csub[8] += -0.3660254037844385*bv;
        __stcg(c + i + 8*ldc, csub[8]);
        csub[10] += 1.366025403784439*bv;
        __stcg(c + i + 10*ldc, csub[10]);
        bv = __ldcg(b + i + 13*ldb);
        csub[9] += -0.3660254037844385*bv;
        __stcg(c + i + 9*ldc, csub[9]);
        csub[11] += 1.366025403784439*bv;
        __stcg(c + i + 11*ldc, csub[11]);
        bv = __ldcg(b + i + 14*ldb);
        csub[12] += -0.3660254037844385*bv;
        __stcg(c + i + 12*ldc, csub[12]);
        csub[14] += 1.366025403784439*bv;
        __stcg(c + i + 14*ldc, csub[14]);
        bv = __ldcg(b + i + 15*ldb);
        csub[13] += -0.3660254037844385*bv;
        __stcg(c + i + 13*ldc, csub[13]);
        csub[15] += 1.366025403784439*bv;
        __stcg(c + i + 15*ldc, csub[15]);
        bv = __ldcg(b + i + 16*ldb);
        csub[0] += -1.366025403784439*bv;
        __stcg(c + i + 0*ldc, csub[0]);
        csub[1] += 0.3660254037844385*bv;
        __stcg(c + i + 1*ldc, csub[1]);
        bv = __ldcg(b + i + 17*ldb);
        csub[2] += -1.366025403784439*bv;
        __stcg(c + i + 2*ldc, csub[2]);
        csub[3] += 0.3660254037844385*bv;
        __stcg(c + i + 3*ldc, csub[3]);
        bv = __ldcg(b + i + 18*ldb);
        csub[4] += -1.366025403784439*bv;
        __stcg(c + i + 4*ldc, csub[4]);
        csub[5] += 0.3660254037844385*bv;
        __stcg(c + i + 5*ldc, csub[5]);
        bv = __ldcg(b + i + 19*ldb);
        csub[6] += -1.366025403784439*bv;
        __stcg(c + i + 6*ldc, csub[6]);
        csub[7] += 0.3660254037844385*bv;
        __stcg(c + i + 7*ldc, csub[7]);
        bv = __ldcg(b + i + 20*ldb);
        csub[16] += -0.3660254037844385*bv;
        __stcg(c + i + 16*ldc, csub[16]);
        csub[20] += 1.366025403784439*bv;
        __stcg(c + i + 20*ldc, csub[20]);
        bv = __ldcg(b + i + 21*ldb);
        csub[17] += -0.3660254037844385*bv;
        __stcg(c + i + 17*ldc, csub[17]);
        csub[21] += 1.366025403784439*bv;
        __stcg(c + i + 21*ldc, csub[21]);
        bv = __ldcg(b + i + 22*ldb);
        csub[18] += -0.3660254037844385*bv;
        __stcg(c + i + 18*ldc, csub[18]);
        csub[22] += 1.366025403784439*bv;
        __stcg(c + i + 22*ldc, csub[22]);
        bv = __ldcg(b + i + 23*ldb);
        csub[19] += -0.3660254037844385*bv;
        __stcg(c + i + 19*ldc, csub[19]);
        csub[23] += 1.366025403784439*bv;
        __stcg(c + i + 23*ldc, csub[23]);

    }
}
