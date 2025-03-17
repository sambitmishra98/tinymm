inline __device__ double make_zero()
{ return 0; }

__global__ void
gimmik_mm_8x24(const double* __restrict__ b, double* __restrict__ c)
{
    const int n = 1000000;
    const int ldb = 1000000;
    const int ldc = 1000000;
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    double cv[12], bv[12], dotp;
    __shared__ double csub[1][24][32];

    if (i >= n)
        return;

    if (threadIdx.y == 0)
    {
        bv[0] = __ldcg(b + i + 0*ldb); 
        bv[4] = __ldcg(b + i + 8*ldb); 
        bv[5] = __ldcg(b + i + 10*ldb); 
        bv[8] = __ldcg(b + i + 16*ldb); 
        bv[10] = __ldcg(b + i + 20*ldb); 
        dotp = (0.8660254037844386*bv[0] + 0.8660254037844386*bv[4] + 0.8660254037844386*bv[5] + 0.8660254037844386*bv[8] + 0.8660254037844386*bv[10]);
        cv[0] = dotp;
        dotp = (-0.8660254037844386*bv[0]);
        csub[0][1][threadIdx.x] = dotp;
        bv[1] = __ldcg(b + i + 2*ldb); 
        bv[9] = __ldcg(b + i + 18*ldb); 
        bv[11] = __ldcg(b + i + 22*ldb); 
        dotp = (0.8660254037844386*bv[1] + -0.8660254037844386*bv[4] + -0.8660254037844386*bv[5] + 0.8660254037844386*bv[9] + 0.8660254037844386*bv[11]);
        cv[1] = dotp;
        dotp = (-0.8660254037844386*bv[1]);
        csub[0][3][threadIdx.x] = dotp;
        bv[2] = __ldcg(b + i + 4*ldb); 
        bv[6] = __ldcg(b + i + 12*ldb); 
        bv[7] = __ldcg(b + i + 14*ldb); 
        dotp = (0.8660254037844386*bv[2] + 0.8660254037844386*bv[6] + 0.8660254037844386*bv[7] + -0.8660254037844386*bv[8] + -0.8660254037844386*bv[10]);
        cv[2] = dotp;
        dotp = (-0.8660254037844386*bv[2]);
        csub[0][5][threadIdx.x] = dotp;
        bv[3] = __ldcg(b + i + 6*ldb); 
        dotp = (0.8660254037844386*bv[3] + -0.8660254037844386*bv[6] + -0.8660254037844386*bv[7] + -0.8660254037844386*bv[9] + -0.8660254037844386*bv[11]);
        cv[3] = dotp;
        dotp = (-0.8660254037844386*bv[3]);
        csub[0][7][threadIdx.x] = dotp;
        __barrier_sync(0);
        dotp = cv[0] + csub[0][0][threadIdx.x];
        __stcg(c + i + 0*ldc, dotp);
        dotp = cv[1] + csub[0][2][threadIdx.x];
        __stcg(c + i + 2*ldc, dotp);
        dotp = cv[2] + csub[0][4][threadIdx.x];
        __stcg(c + i + 4*ldc, dotp);
        dotp = cv[3] + csub[0][6][threadIdx.x];
        __stcg(c + i + 6*ldc, dotp);
        __barrier_sync(0);
    }
    if (threadIdx.y == 1)
    {
        bv[0] = __ldcg(b + i + 1*ldb); 
        dotp = (0.8660254037844386*bv[0]);
        csub[0][0][threadIdx.x] = dotp;
        bv[4] = __ldcg(b + i + 9*ldb); 
        bv[5] = __ldcg(b + i + 11*ldb); 
        bv[8] = __ldcg(b + i + 17*ldb); 
        bv[10] = __ldcg(b + i + 21*ldb); 
        dotp = (-0.8660254037844386*bv[0] + 0.8660254037844386*bv[4] + 0.8660254037844386*bv[5] + 0.8660254037844386*bv[8] + 0.8660254037844386*bv[10]);
        cv[0] = dotp;
        bv[1] = __ldcg(b + i + 3*ldb); 
        dotp = (0.8660254037844386*bv[1]);
        csub[0][2][threadIdx.x] = dotp;
        bv[9] = __ldcg(b + i + 19*ldb); 
        bv[11] = __ldcg(b + i + 23*ldb); 
        dotp = (-0.8660254037844386*bv[1] + -0.8660254037844386*bv[4] + -0.8660254037844386*bv[5] + 0.8660254037844386*bv[9] + 0.8660254037844386*bv[11]);
        cv[1] = dotp;
        bv[2] = __ldcg(b + i + 5*ldb); 
        dotp = (0.8660254037844386*bv[2]);
        csub[0][4][threadIdx.x] = dotp;
        bv[6] = __ldcg(b + i + 13*ldb); 
        bv[7] = __ldcg(b + i + 15*ldb); 
        dotp = (-0.8660254037844386*bv[2] + 0.8660254037844386*bv[6] + 0.8660254037844386*bv[7] + -0.8660254037844386*bv[8] + -0.8660254037844386*bv[10]);
        cv[2] = dotp;
        bv[3] = __ldcg(b + i + 7*ldb); 
        dotp = (0.8660254037844386*bv[3]);
        csub[0][6][threadIdx.x] = dotp;
        dotp = (-0.8660254037844386*bv[3] + -0.8660254037844386*bv[6] + -0.8660254037844386*bv[7] + -0.8660254037844386*bv[9] + -0.8660254037844386*bv[11]);
        cv[3] = dotp;
        __barrier_sync(0);
        dotp = cv[0] + csub[0][1][threadIdx.x];
        __stcg(c + i + 1*ldc, dotp);
        dotp = cv[1] + csub[0][3][threadIdx.x];
        __stcg(c + i + 3*ldc, dotp);
        dotp = cv[2] + csub[0][5][threadIdx.x];
        __stcg(c + i + 5*ldc, dotp);
        dotp = cv[3] + csub[0][7][threadIdx.x];
        __stcg(c + i + 7*ldc, dotp);
        __barrier_sync(0);
    }
}
