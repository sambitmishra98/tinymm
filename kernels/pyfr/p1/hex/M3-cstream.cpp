inline __device__ double make_zero()
{ return 0; }

__global__ void
gimmik_mm_8x24(const double* __restrict__ b, double* __restrict__ c)
{
    const int n = 1000000;
    const int ldb = 1000000;
    const int ldc = 1000000;
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i < n)
    {
        double bv, csub[8];

        bv = __ldcg(b + i + 0*ldb);
        csub[0] = 1.366025403784439*bv;
        csub[4] = -0.3660254037844385*bv;
        bv = __ldcg(b + i + 1*ldb);
        csub[1] = 1.366025403784439*bv;
        csub[5] = -0.3660254037844385*bv;
        bv = __ldcg(b + i + 2*ldb);
        csub[2] = 1.366025403784439*bv;
        csub[6] = -0.3660254037844385*bv;
        bv = __ldcg(b + i + 3*ldb);
        csub[3] = 1.366025403784439*bv;
        csub[7] = -0.3660254037844385*bv;
        bv = __ldcg(b + i + 4*ldb);
        csub[0] += 1.366025403784439*bv;
        csub[2] += -0.3660254037844385*bv;
        bv = __ldcg(b + i + 5*ldb);
        csub[1] += 1.366025403784439*bv;
        csub[3] += -0.3660254037844385*bv;
        bv = __ldcg(b + i + 6*ldb);
        csub[4] += 1.366025403784439*bv;
        csub[6] += -0.3660254037844385*bv;
        bv = __ldcg(b + i + 7*ldb);
        csub[5] += 1.366025403784439*bv;
        csub[7] += -0.3660254037844385*bv;
        bv = __ldcg(b + i + 8*ldb);
        csub[0] += -0.3660254037844385*bv;
        csub[1] += 1.366025403784439*bv;
        bv = __ldcg(b + i + 9*ldb);
        csub[2] += -0.3660254037844385*bv;
        csub[3] += 1.366025403784439*bv;
        bv = __ldcg(b + i + 10*ldb);
        csub[4] += -0.3660254037844385*bv;
        csub[5] += 1.366025403784439*bv;
        bv = __ldcg(b + i + 11*ldb);
        csub[6] += -0.3660254037844385*bv;
        csub[7] += 1.366025403784439*bv;
        bv = __ldcg(b + i + 12*ldb);
        csub[0] += -0.3660254037844385*bv;
        csub[2] += 1.366025403784439*bv;
        bv = __ldcg(b + i + 13*ldb);
        csub[1] += -0.3660254037844385*bv;
        csub[3] += 1.366025403784439*bv;
        bv = __ldcg(b + i + 14*ldb);
        csub[4] += -0.3660254037844385*bv;
        csub[6] += 1.366025403784439*bv;
        bv = __ldcg(b + i + 15*ldb);
        csub[5] += -0.3660254037844385*bv;
        csub[7] += 1.366025403784439*bv;
        bv = __ldcg(b + i + 16*ldb);
        csub[0] += 1.366025403784439*bv;
        csub[1] += -0.3660254037844385*bv;
        bv = __ldcg(b + i + 17*ldb);
        csub[2] += 1.366025403784439*bv;
        csub[3] += -0.3660254037844385*bv;
        bv = __ldcg(b + i + 18*ldb);
        csub[4] += 1.366025403784439*bv;
        csub[5] += -0.3660254037844385*bv;
        bv = __ldcg(b + i + 19*ldb);
        csub[6] += 1.366025403784439*bv;
        csub[7] += -0.3660254037844385*bv;
        bv = __ldcg(b + i + 20*ldb);
        csub[0] += -0.3660254037844385*bv;
        __stcg(c + i + 0*ldc, csub[0]);
        csub[4] += 1.366025403784439*bv;
        __stcg(c + i + 4*ldc, csub[4]);
        bv = __ldcg(b + i + 21*ldb);
        csub[1] += -0.3660254037844385*bv;
        __stcg(c + i + 1*ldc, csub[1]);
        csub[5] += 1.366025403784439*bv;
        __stcg(c + i + 5*ldc, csub[5]);
        bv = __ldcg(b + i + 22*ldb);
        csub[2] += -0.3660254037844385*bv;
        __stcg(c + i + 2*ldc, csub[2]);
        csub[6] += 1.366025403784439*bv;
        __stcg(c + i + 6*ldc, csub[6]);
        bv = __ldcg(b + i + 23*ldb);
        csub[3] += -0.3660254037844385*bv;
        __stcg(c + i + 3*ldc, csub[3]);
        csub[7] += 1.366025403784439*bv;
        __stcg(c + i + 7*ldc, csub[7]);

    }
}
