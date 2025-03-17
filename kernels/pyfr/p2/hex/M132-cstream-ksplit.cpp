inline __device__ double make_zero()
{ return 0; }

__global__ void
gimmik_mm_27x81(const double* __restrict__ b, double* __restrict__ c)
{
    const int n = 1000000;
    const int ldb = 1000000;
    const int ldc = 1000000;
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    double cv[12], bv[41], dotp;
    __shared__ double csub[1][24][32];

    if (i >= n)
        return;

    if (threadIdx.y == 0)
    {
        bv[0] = __ldcg(b + i + 0*ldb); 
        bv[1] = __ldcg(b + i + 2*ldb); 
        bv[15] = __ldcg(b + i + 30*ldb); 
        bv[27] = __ldcg(b + i + 54*ldb); 
        bv[36] = __ldcg(b + i + 72*ldb); 
        dotp = (1.936491673103709*bv[0] + -0.6454972243679035*bv[1] + 1.032795558988645*bv[15] + 1.936491673103709*bv[27] + -0.6454972243679035*bv[36]);
        cv[0] = dotp;
        bv[14] = __ldcg(b + i + 28*ldb); 
        bv[17] = __ldcg(b + i + 34*ldb); 
        bv[32] = __ldcg(b + i + 64*ldb); 
        dotp = (-1.613743060919757*bv[0] + 1.613743060919757*bv[1] + 1.936491673103709*bv[14] + -0.6454972243679035*bv[17] + 1.032795558988645*bv[32]);
        csub[0][1][threadIdx.x] = dotp;
        bv[16] = __ldcg(b + i + 32*ldb); 
        bv[28] = __ldcg(b + i + 56*ldb); 
        bv[37] = __ldcg(b + i + 74*ldb); 
        dotp = (0.6454972243679035*bv[0] + -1.936491673103709*bv[1] + 1.032795558988645*bv[16] + 1.936491673103709*bv[28] + -0.6454972243679035*bv[37]);
        cv[1] = dotp;
        bv[2] = __ldcg(b + i + 4*ldb); 
        bv[33] = __ldcg(b + i + 66*ldb); 
        dotp = (1.032795558988645*bv[2] + 1.032795558988645*bv[33]);
        csub[0][3][threadIdx.x] = dotp;
        bv[29] = __ldcg(b + i + 58*ldb); 
        bv[38] = __ldcg(b + i + 76*ldb); 
        dotp = (-1.613743060919757*bv[14] + 1.613743060919757*bv[17] + 1.936491673103709*bv[29] + -0.6454972243679035*bv[38]);
        cv[2] = dotp;
        bv[34] = __ldcg(b + i + 68*ldb); 
        dotp = (-1.032795558988645*bv[2] + 1.032795558988645*bv[34]);
        csub[0][5][threadIdx.x] = dotp;
        bv[3] = __ldcg(b + i + 6*ldb); 
        bv[4] = __ldcg(b + i + 8*ldb); 
        bv[30] = __ldcg(b + i + 60*ldb); 
        bv[39] = __ldcg(b + i + 78*ldb); 
        dotp = (1.936491673103709*bv[3] + -0.6454972243679035*bv[4] + -1.032795558988645*bv[15] + 1.936491673103709*bv[30] + -0.6454972243679035*bv[39]);
        cv[3] = dotp;
        bv[35] = __ldcg(b + i + 70*ldb); 
        dotp = (-1.613743060919757*bv[3] + 1.613743060919757*bv[4] + 0.6454972243679035*bv[14] + -1.936491673103709*bv[17] + 1.032795558988645*bv[35]);
        csub[0][7][threadIdx.x] = dotp;
        bv[31] = __ldcg(b + i + 62*ldb); 
        bv[40] = __ldcg(b + i + 80*ldb); 
        dotp = (0.6454972243679035*bv[3] + -1.936491673103709*bv[4] + -1.032795558988645*bv[16] + 1.936491673103709*bv[31] + -0.6454972243679035*bv[40]);
        cv[4] = dotp;
        bv[5] = __ldcg(b + i + 10*ldb); 
        bv[18] = __ldcg(b + i + 36*ldb); 
        bv[21] = __ldcg(b + i + 42*ldb); 
        dotp = (1.032795558988645*bv[5] + 1.936491673103709*bv[18] + -0.6454972243679035*bv[21] + -1.613743060919757*bv[27] + 1.613743060919757*bv[36]);
        csub[0][9][threadIdx.x] = dotp;
        bv[20] = __ldcg(b + i + 40*ldb); 
        dotp = (1.032795558988645*bv[20]);
        cv[5] = dotp;
        bv[19] = __ldcg(b + i + 38*ldb); 
        bv[22] = __ldcg(b + i + 44*ldb); 
        dotp = (-1.032795558988645*bv[5] + 1.936491673103709*bv[19] + -0.6454972243679035*bv[22] + -1.613743060919757*bv[28] + 1.613743060919757*bv[37]);
        csub[0][11][threadIdx.x] = dotp;
        bv[6] = __ldcg(b + i + 12*ldb); 
        bv[7] = __ldcg(b + i + 14*ldb); 
        dotp = (1.936491673103709*bv[6] + -0.6454972243679035*bv[7] + -1.613743060919757*bv[18] + 1.613743060919757*bv[21]);
        cv[6] = dotp;
        dotp = (-1.613743060919757*bv[6] + 1.613743060919757*bv[7] + -1.613743060919757*bv[29] + 1.613743060919757*bv[38]);
        csub[0][13][threadIdx.x] = dotp;
        dotp = (0.6454972243679035*bv[6] + -1.936491673103709*bv[7] + -1.613743060919757*bv[19] + 1.613743060919757*bv[22]);
        cv[7] = dotp;
        bv[8] = __ldcg(b + i + 16*ldb); 
        dotp = (1.032795558988645*bv[8] + 0.6454972243679035*bv[18] + -1.936491673103709*bv[21] + -1.613743060919757*bv[30] + 1.613743060919757*bv[39]);
        csub[0][15][threadIdx.x] = dotp;
        dotp = (-1.032795558988645*bv[20]);
        cv[8] = dotp;
        dotp = (-1.032795558988645*bv[8] + 0.6454972243679035*bv[19] + -1.936491673103709*bv[22] + -1.613743060919757*bv[31] + 1.613743060919757*bv[40]);
        csub[0][17][threadIdx.x] = dotp;
        bv[9] = __ldcg(b + i + 18*ldb); 
        bv[10] = __ldcg(b + i + 20*ldb); 
        bv[24] = __ldcg(b + i + 48*ldb); 
        dotp = (1.936491673103709*bv[9] + -0.6454972243679035*bv[10] + 1.032795558988645*bv[24] + 0.6454972243679035*bv[27] + -1.936491673103709*bv[36]);
        cv[9] = dotp;
        bv[23] = __ldcg(b + i + 46*ldb); 
        bv[26] = __ldcg(b + i + 52*ldb); 
        dotp = (-1.613743060919757*bv[9] + 1.613743060919757*bv[10] + 1.936491673103709*bv[23] + -0.6454972243679035*bv[26] + -1.032795558988645*bv[32]);
        csub[0][19][threadIdx.x] = dotp;
        bv[25] = __ldcg(b + i + 50*ldb); 
        dotp = (0.6454972243679035*bv[9] + -1.936491673103709*bv[10] + 1.032795558988645*bv[25] + 0.6454972243679035*bv[28] + -1.936491673103709*bv[37]);
        cv[10] = dotp;
        bv[11] = __ldcg(b + i + 22*ldb); 
        dotp = (1.032795558988645*bv[11] + -1.032795558988645*bv[33]);
        csub[0][21][threadIdx.x] = dotp;
        dotp = (-1.613743060919757*bv[23] + 1.613743060919757*bv[26] + 0.6454972243679035*bv[29] + -1.936491673103709*bv[38]);
        cv[11] = dotp;
        dotp = (-1.032795558988645*bv[11] + -1.032795558988645*bv[34]);
        csub[0][23][threadIdx.x] = dotp;
        __barrier_sync(0);
        dotp = cv[0] + csub[0][0][threadIdx.x];
        __stcg(c + i + 0*ldc, dotp);
        dotp = cv[1] + csub[0][2][threadIdx.x];
        __stcg(c + i + 2*ldc, dotp);
        dotp = cv[2] + csub[0][4][threadIdx.x];
        __stcg(c + i + 4*ldc, dotp);
        dotp = cv[3] + csub[0][6][threadIdx.x];
        __stcg(c + i + 6*ldc, dotp);
        dotp = cv[4] + csub[0][8][threadIdx.x];
        __stcg(c + i + 8*ldc, dotp);
        dotp = cv[5] + csub[0][10][threadIdx.x];
        __stcg(c + i + 10*ldc, dotp);
        dotp = cv[6] + csub[0][12][threadIdx.x];
        __stcg(c + i + 12*ldc, dotp);
        dotp = cv[7] + csub[0][14][threadIdx.x];
        __stcg(c + i + 14*ldc, dotp);
        dotp = cv[8] + csub[0][16][threadIdx.x];
        __stcg(c + i + 16*ldc, dotp);
        dotp = cv[9] + csub[0][18][threadIdx.x];
        __stcg(c + i + 18*ldc, dotp);
        dotp = cv[10] + csub[0][20][threadIdx.x];
        __stcg(c + i + 20*ldc, dotp);
        dotp = cv[11] + csub[0][22][threadIdx.x];
        __stcg(c + i + 22*ldc, dotp);
        __barrier_sync(0);
        bv[12] = __ldcg(b + i + 24*ldb); 
        bv[13] = __ldcg(b + i + 26*ldb); 
        dotp = (1.936491673103709*bv[12] + -0.6454972243679035*bv[13] + -1.032795558988645*bv[24] + 0.6454972243679035*bv[30] + -1.936491673103709*bv[39]);
        cv[0] = dotp;
        dotp = (-1.613743060919757*bv[12] + 1.613743060919757*bv[13] + 0.6454972243679035*bv[23] + -1.936491673103709*bv[26] + -1.032795558988645*bv[35]);
        csub[0][1][threadIdx.x] = dotp;
        dotp = (0.6454972243679035*bv[12] + -1.936491673103709*bv[13] + -1.032795558988645*bv[25] + 0.6454972243679035*bv[31] + -1.936491673103709*bv[40]);
        cv[1] = dotp;
        __barrier_sync(0);
        dotp = cv[0] + csub[0][0][threadIdx.x];
        __stcg(c + i + 24*ldc, dotp);
        dotp = cv[1] + csub[0][2][threadIdx.x];
        __stcg(c + i + 26*ldc, dotp);
        __barrier_sync(0);
    }
    if (threadIdx.y == 1)
    {
        bv[0] = __ldcg(b + i + 1*ldb); 
        bv[13] = __ldcg(b + i + 27*ldb); 
        bv[16] = __ldcg(b + i + 33*ldb); 
        bv[31] = __ldcg(b + i + 63*ldb); 
        dotp = (1.032795558988645*bv[0] + 1.936491673103709*bv[13] + -0.6454972243679035*bv[16] + 1.032795558988645*bv[31]);
        csub[0][0][threadIdx.x] = dotp;
        bv[15] = __ldcg(b + i + 31*ldb); 
        bv[27] = __ldcg(b + i + 55*ldb); 
        bv[36] = __ldcg(b + i + 73*ldb); 
        dotp = (1.032795558988645*bv[15] + 1.936491673103709*bv[27] + -0.6454972243679035*bv[36]);
        cv[0] = dotp;
        bv[14] = __ldcg(b + i + 29*ldb); 
        bv[17] = __ldcg(b + i + 35*ldb); 
        bv[32] = __ldcg(b + i + 65*ldb); 
        dotp = (-1.032795558988645*bv[0] + 1.936491673103709*bv[14] + -0.6454972243679035*bv[17] + 1.032795558988645*bv[32]);
        csub[0][2][threadIdx.x] = dotp;
        bv[1] = __ldcg(b + i + 3*ldb); 
        bv[2] = __ldcg(b + i + 5*ldb); 
        bv[28] = __ldcg(b + i + 57*ldb); 
        bv[37] = __ldcg(b + i + 75*ldb); 
        dotp = (1.936491673103709*bv[1] + -0.6454972243679035*bv[2] + -1.613743060919757*bv[13] + 1.613743060919757*bv[16] + 1.936491673103709*bv[28] + -0.6454972243679035*bv[37]);
        cv[1] = dotp;
        bv[33] = __ldcg(b + i + 67*ldb); 
        dotp = (-1.613743060919757*bv[1] + 1.613743060919757*bv[2] + 1.032795558988645*bv[33]);
        csub[0][4][threadIdx.x] = dotp;
        bv[29] = __ldcg(b + i + 59*ldb); 
        bv[38] = __ldcg(b + i + 77*ldb); 
        dotp = (0.6454972243679035*bv[1] + -1.936491673103709*bv[2] + -1.613743060919757*bv[14] + 1.613743060919757*bv[17] + 1.936491673103709*bv[29] + -0.6454972243679035*bv[38]);
        cv[2] = dotp;
        bv[3] = __ldcg(b + i + 7*ldb); 
        bv[34] = __ldcg(b + i + 69*ldb); 
        dotp = (1.032795558988645*bv[3] + 0.6454972243679035*bv[13] + -1.936491673103709*bv[16] + 1.032795558988645*bv[34]);
        csub[0][6][threadIdx.x] = dotp;
        bv[30] = __ldcg(b + i + 61*ldb); 
        bv[39] = __ldcg(b + i + 79*ldb); 
        dotp = (-1.032795558988645*bv[15] + 1.936491673103709*bv[30] + -0.6454972243679035*bv[39]);
        cv[3] = dotp;
        bv[35] = __ldcg(b + i + 71*ldb); 
        dotp = (-1.032795558988645*bv[3] + 0.6454972243679035*bv[14] + -1.936491673103709*bv[17] + 1.032795558988645*bv[35]);
        csub[0][8][threadIdx.x] = dotp;
        bv[4] = __ldcg(b + i + 9*ldb); 
        bv[5] = __ldcg(b + i + 11*ldb); 
        bv[19] = __ldcg(b + i + 39*ldb); 
        dotp = (1.936491673103709*bv[4] + -0.6454972243679035*bv[5] + 1.032795558988645*bv[19]);
        cv[4] = dotp;
        bv[18] = __ldcg(b + i + 37*ldb); 
        bv[21] = __ldcg(b + i + 43*ldb); 
        dotp = (-1.613743060919757*bv[4] + 1.613743060919757*bv[5] + 1.936491673103709*bv[18] + -0.6454972243679035*bv[21] + -1.613743060919757*bv[27] + 1.613743060919757*bv[36]);
        csub[0][10][threadIdx.x] = dotp;
        bv[20] = __ldcg(b + i + 41*ldb); 
        dotp = (0.6454972243679035*bv[4] + -1.936491673103709*bv[5] + 1.032795558988645*bv[20]);
        cv[5] = dotp;
        bv[6] = __ldcg(b + i + 13*ldb); 
        dotp = (1.032795558988645*bv[6] + -1.613743060919757*bv[28] + 1.613743060919757*bv[37]);
        csub[0][12][threadIdx.x] = dotp;
        dotp = (-1.613743060919757*bv[18] + 1.613743060919757*bv[21]);
        cv[6] = dotp;
        dotp = (-1.032795558988645*bv[6] + -1.613743060919757*bv[29] + 1.613743060919757*bv[38]);
        csub[0][14][threadIdx.x] = dotp;
        bv[7] = __ldcg(b + i + 15*ldb); 
        bv[8] = __ldcg(b + i + 17*ldb); 
        dotp = (1.936491673103709*bv[7] + -0.6454972243679035*bv[8] + -1.032795558988645*bv[19]);
        cv[7] = dotp;
        dotp = (-1.613743060919757*bv[7] + 1.613743060919757*bv[8] + 0.6454972243679035*bv[18] + -1.936491673103709*bv[21] + -1.613743060919757*bv[30] + 1.613743060919757*bv[39]);
        csub[0][16][threadIdx.x] = dotp;
        dotp = (0.6454972243679035*bv[7] + -1.936491673103709*bv[8] + -1.032795558988645*bv[20]);
        cv[8] = dotp;
        bv[9] = __ldcg(b + i + 19*ldb); 
        bv[22] = __ldcg(b + i + 45*ldb); 
        bv[25] = __ldcg(b + i + 51*ldb); 
        dotp = (1.032795558988645*bv[9] + 1.936491673103709*bv[22] + -0.6454972243679035*bv[25] + -1.032795558988645*bv[31]);
        csub[0][18][threadIdx.x] = dotp;
        bv[24] = __ldcg(b + i + 49*ldb); 
        dotp = (1.032795558988645*bv[24] + 0.6454972243679035*bv[27] + -1.936491673103709*bv[36]);
        cv[9] = dotp;
        bv[23] = __ldcg(b + i + 47*ldb); 
        bv[26] = __ldcg(b + i + 53*ldb); 
        dotp = (-1.032795558988645*bv[9] + 1.936491673103709*bv[23] + -0.6454972243679035*bv[26] + -1.032795558988645*bv[32]);
        csub[0][20][threadIdx.x] = dotp;
        bv[10] = __ldcg(b + i + 21*ldb); 
        bv[11] = __ldcg(b + i + 23*ldb); 
        dotp = (1.936491673103709*bv[10] + -0.6454972243679035*bv[11] + -1.613743060919757*bv[22] + 1.613743060919757*bv[25] + 0.6454972243679035*bv[28] + -1.936491673103709*bv[37]);
        cv[10] = dotp;
        dotp = (-1.613743060919757*bv[10] + 1.613743060919757*bv[11] + -1.032795558988645*bv[33]);
        csub[0][22][threadIdx.x] = dotp;
        dotp = (0.6454972243679035*bv[10] + -1.936491673103709*bv[11] + -1.613743060919757*bv[23] + 1.613743060919757*bv[26] + 0.6454972243679035*bv[29] + -1.936491673103709*bv[38]);
        cv[11] = dotp;
        __barrier_sync(0);
        dotp = cv[0] + csub[0][1][threadIdx.x];
        __stcg(c + i + 1*ldc, dotp);
        dotp = cv[1] + csub[0][3][threadIdx.x];
        __stcg(c + i + 3*ldc, dotp);
        dotp = cv[2] + csub[0][5][threadIdx.x];
        __stcg(c + i + 5*ldc, dotp);
        dotp = cv[3] + csub[0][7][threadIdx.x];
        __stcg(c + i + 7*ldc, dotp);
        dotp = cv[4] + csub[0][9][threadIdx.x];
        __stcg(c + i + 9*ldc, dotp);
        dotp = cv[5] + csub[0][11][threadIdx.x];
        __stcg(c + i + 11*ldc, dotp);
        dotp = cv[6] + csub[0][13][threadIdx.x];
        __stcg(c + i + 13*ldc, dotp);
        dotp = cv[7] + csub[0][15][threadIdx.x];
        __stcg(c + i + 15*ldc, dotp);
        dotp = cv[8] + csub[0][17][threadIdx.x];
        __stcg(c + i + 17*ldc, dotp);
        dotp = cv[9] + csub[0][19][threadIdx.x];
        __stcg(c + i + 19*ldc, dotp);
        dotp = cv[10] + csub[0][21][threadIdx.x];
        __stcg(c + i + 21*ldc, dotp);
        dotp = cv[11] + csub[0][23][threadIdx.x];
        __stcg(c + i + 23*ldc, dotp);
        __barrier_sync(0);
        bv[12] = __ldcg(b + i + 25*ldb); 
        dotp = (1.032795558988645*bv[12] + 0.6454972243679035*bv[22] + -1.936491673103709*bv[25] + -1.032795558988645*bv[34]);
        csub[0][0][threadIdx.x] = dotp;
        dotp = (-1.032795558988645*bv[24] + 0.6454972243679035*bv[30] + -1.936491673103709*bv[39]);
        cv[0] = dotp;
        dotp = (-1.032795558988645*bv[12] + 0.6454972243679035*bv[23] + -1.936491673103709*bv[26] + -1.032795558988645*bv[35]);
        csub[0][2][threadIdx.x] = dotp;
        __barrier_sync(0);
        dotp = cv[0] + csub[0][1][threadIdx.x];
        __stcg(c + i + 25*ldc, dotp);
        __barrier_sync(0);
    }
}
