/*
 * CPQREF/cuda_pqntrusign.cu
 *
 *  Copyright 2014 John M. Schanck
 *
 *  This file is part of CPQREF.
 *
 *  CPQREF is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  CPQREF is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with CPQREF.  If not, see <http://www.gnu.org/licenses/>.
*/

extern "C" {

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "shred.h"
#include "params.h"
#include "pack.h"
#include "pol.h"
#include "pqerror.h"
#include "pqntrusign.h"
#include "randombytes.h"

//#define CHECK_NORM_AF // check ||a*f||
//#define CHECK_NORM_AG // check ||a*g||
#define CHECK_NORM_S // check ||s0+a*f||
#define CHECK_NORM_T // check ||t0+a*g||

#define STATS 1

//extern unsigned long int attempts;
//extern double t_prep;
//extern double t_dowhile;
//extern double t_writeback;

/** Multiple-GPU mode */
#define	nGPUs	1	// single GPU for now

/** Data type of R(q/2) coefficients */
#define	RqMode	32	// 32 or 64
#if RqMode == 32
typedef	int32_t	int_q;
#elif RqMode == 64
typedef	int64_t	int_q;
#endif

/** Data type of R(p/2) coefficients */
typedef	int8_t	int_p;

/** CUDA runtime error detector */
#define		CSC(err)	__cudaSafeCall(err, __FILE__, __LINE__)
#define		CCE()		__cudaCheckError(__FILE__, __LINE__)
inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
	if (cudaSuccess != err) {
		fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}
	return;
}
inline void __cudaCheckError(const char *file, const int line) {
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}
	return;
}

/** Texture memory */
texture<int_p, 1> tex_sp;
texture<int_p, 1> tex_tp;
texture<int_p, 1> tex_ginv;
texture<int_q, 1> tex_h;
//texture<uint16_t, 1> tex_F;
//texture<uint16_t, 1> tex_G;
texture<unsigned, 1> tex_rng;

__constant__ uint16_t con_F[1024];
__constant__ uint16_t con_G[1024];

/** Definitions */
#define		salsa20_rounds			20
#define		salsa20_key_bytes		32
#define		salsa20_nonce_bytes		8

/** CUDA inline device functions */
__inline__ __device__
unsigned rotate(unsigned u, int c) {
	return (u<<c) | (u>>(32-c));
}

__inline__ __device__
void rotate4(unsigned *x, int offset0, int offset1, int offset2, int offset3) {
	x[offset0] ^= rotate(x[offset3]+x[offset2], 7);
	x[offset1] ^= rotate(x[offset0]+x[offset3], 9);
	x[offset2] ^= rotate(x[offset1]+x[offset0], 13);
	x[offset3] ^= rotate(x[offset2]+x[offset1], 18);
}

__inline__ __device__
void salsa20(unsigned *x) {
#pragma unroll
	for (int i=salsa20_rounds; i>0; i-=2) {
		rotate4(x, 4, 8, 12, 0);
		rotate4(x, 9, 13, 1, 5);
		rotate4(x, 14, 2, 6, 10);
		rotate4(x, 3, 7, 11, 15);
		rotate4(x, 1, 2, 3, 0);
		rotate4(x, 6, 7, 4, 5);
		rotate4(x, 11, 8, 9, 10);
		rotate4(x, 12, 13, 14, 15);
	}
}

#define		tid		(threadIdx.y*blockDim.x+threadIdx.x)

__inline__ __device__
int_q RqRq(int_q *A, int_q *B, int NN, int qq, int qq_) {
	int64_t t = 0;
	for (int i=0; i<NN; i++)
		t += A[i]*B[(tid-i+NN)%NN];
	t &= qq_;
	if (t < 0)
		t += qq;
	if (t > qq_/2)
		t -= qq;
	return (int_q)t;
}

__inline__ __device__
int_p RpRp(int_p *A, int_p *B, int NN, int pp) {
	int32_t t = 0;
	for (int i=0; i<NN; i++)
		t += A[i]*B[(tid-i+NN)%NN];
	t %= pp;
	if (t < 0)
		t += pp;
	if (t > (pp-1)/2)
		t -= pp;
	return (int_p)t;
}

__inline__ __device__
int_q RpRppp(int_q *T, int_p *A, uint16_t *B3, int NN, int d0, int d1, int d2) {
	uint16_t *tptr = B3;
	int_q t = 0;
	for (int j=0; j<d0; j++) {
		t += A[(tid-tptr[0]+NN)%NN];
		t -= A[(tid-tptr[d0]+NN)%NN];
		tptr ++;
	}
	T[tid] = t;
	__syncthreads();
	t = 0;
	tptr += d0;
	for (int j=0; j<d1; j++) {
		t += T[(tid-tptr[0]+NN)%NN];
		t -= T[(tid-tptr[d1]+NN)%NN];
		tptr ++;
	}
	tptr += d1;
	for (int j=0; j<d2; j++) {
		t += A[(tid-tptr[0]+NN)%NN];
		t -= A[(tid-tptr[d2]+NN)%NN];
		tptr ++;
	}
	return (int_q)t;
}

/** CUDA Kernel **
 *	Each block tries to generate one signature (valid or not).
 *	Offset/position of the valid signatures is marked in bit stream "pos".
 *	The number of valid signatures generated is saved in "numValid[0]".
 *	If the i-th bit of "pos" is "1", "&s[i*N]" holds a valid signature.
 */
__global__ void cuda_sign(
		int_q *s,
		int *pos,
		int *numValid,
		int N,
		int_q q,
		int_p p,
		int d0,
		int d1,
		int d2,
		int_q norm_bound_s,
		int_q norm_bound_t,
		int_p *d_tp)
{
	//// Shorthands
	int dd = 2*(d0+d1+d2); // length of product form polynomial
	unsigned long inc = blockIdx.x*blockDim.y+threadIdx.y; // nonce-increment
	int_q center = (q+p-1)/(2*p)-1;
	int_q range = 2*center+1;
	uint32_t cap = (uint32_t)0xffffffff-(uint32_t)0xffffffff%range;
	//// Shared memory: from large data types to small types
	extern __shared__ char buff[];
	int_q		*s0		= (int_q *)buff; // int_q[N]
	int_q		*t0		= (int_q *)(&s0[N]); // int_q[N]
	int_q		*h		= (int_q *)(&t0[N]); // int_q[N]
	unsigned	*rng	= (unsigned *)(&h[N]); // unsigned[blockDim.x*blockDim.y]
	int			*valid	= (int *)(&rng[blockDim.x*blockDim.y]); // int[1]
	uint16_t	*F	= (uint16_t *)(&valid[1]); // int16_t[dd]
	uint16_t	*G	= (uint16_t *)(&F[dd]); // int16_t[dd]
	int_p		*a		= (int_p *)(&G[dd]); // int_p[N]
	int_p		*ginv	= (int_p *)(&a[N]); // int_p[N]
	int_p		*tp_t0	= (int_p *)(&ginv[N]); // int_p[N]

	//// Load global/texture memory to shared memory
	if (tid == 0)
		valid[0] = 1;
	rng[tid] = (unsigned)tex1Dfetch(tex_rng, threadIdx.x);
	if (threadIdx.x == 8)
		rng[tid] += (unsigned)inc;
	if (threadIdx.x == 9)
		rng[tid] += (unsigned)(inc>>32);
	__syncthreads(); // sync: rng, valid

	//// RNG: uniformly generates one random pZ <- Z(q/2)
	unsigned rng_temp[16];
	for (int i=0; i<16; i++)
		rng_temp[i] = rng[blockDim.x*threadIdx.y+i];
	// salsa stream cipher
	salsa20(rng_temp);

	if (tid < N) {
		int_p temp_sp = (int_p)tex1Dfetch(tex_sp, tid);
		uint32_t r = rng_temp[threadIdx.x] + rng[tid];
		// if not uniform, set as invalid
		if (r >= cap)
			atomicExch(valid, 0);
		// s0=sp+p*rand
		s0[tid] = ((int_q)(r%range)-center)*p+temp_sp;
		h[tid] = (int_q)tex1Dfetch(tex_h, tid); // h <- R(q/2)
		__syncthreads(); // sync: s0, h

		//// compute signature from s0=sp+p*rand <- R(q/2)
		// t0=h*s0 <- R(q/2)
		t0[tid] = RqRq(h, s0, N, q, q-1);
		// tp_t0 = tp-t0 <- R(p/2)
		tp_t0[tid] = ((int_q)(int_p)tex1Dfetch(tex_tp, tid)-t0[tid])%p;
		if (tp_t0[tid] < 0)
			tp_t0[tid] += p;
		if (tp_t0[tid] > (p-1)/2)
			tp_t0[tid] -= p;
		ginv[tid] = (int_p)tex1Dfetch(tex_ginv, tid);// ginv <- R(p/2)
		__syncthreads(); // sync: tp_t0, ginv

		// a=(tp-t0)*ginv <- R(p/2)
		a[tid] = RpRp(ginv, tp_t0, N, p);
//		if (tid < dd) {
//			F[tid] = (uint16_t)tex1Dfetch(tex_F, tid); // F=f/p-1
//			G[tid] = (uint16_t)tex1Dfetch(tex_G, tid); // G=g-1
//		}
		__syncthreads(); // sync: a, F, G

		//// check norm
		int_q norm = 0;

		//////////////////////////////////////
		//// the code below checks validity
		//// t = a * G + t0
		if (valid[0] == 1) { // todo: maybe remove if
			norm = RpRppp(h, a, con_G, N, d0, d1, d2)+a[tid];
#ifdef CHECK_NORM_AG
			// check ||a*g||
			if (norm > q/2-norm_bound_t || -norm > q/2-norm_bound_t)
				atomicExch(valid, 0);
			// end of check ||a*g||
#endif
			norm += t0[tid];
#ifdef CHECK_NORM_T
			// check ||t||
			if (norm > norm_bound_t || -norm > norm_bound_t)
				atomicExch(valid, 0);
			// end of check ||t||
#endif
		}
		//// s = a * f + s0
		if (valid[0] == 1) { // todo: maybe remove if
			norm = (RpRppp(t0, a, con_F, N, d0, d1, d2)+a[tid])*p;
#ifdef CHECK_NORM_AF
			// check ||a*f||
			if (norm > q/2-norm_bound_s || -norm > q/2-norm_bound_s)
				atomicExch(valid, 0);
			// end of check ||a*f||
#endif
			norm += s0[tid];
#ifdef CHECK_NORM_S
			// check ||s||
			if (norm > norm_bound_s || -norm > norm_bound_s)
				atomicExch(valid, 0);
			// end of check ||s||
#endif
		}
		__syncthreads(); // sync: valid
		//// end of checking ||(s, t)||

		//// write back
		if (valid[0] == 1) { // avoid unnecessary global memory store
			if (tid == 0) {
				atomicOr(&pos[blockIdx.x/32], 0x1<<(blockIdx.x%32)); // mark position
				atomicAdd(&numValid[0], 1); // accumulate valid count
			}
			norm -= temp_sp;
			norm /= p;
			norm += q/(2*p);
			s[blockIdx.x*blockDim.x*blockDim.y+tid] = norm;
		}
	}
}

/** CUDA pqntrusign */
const char *sigma = "expand 32-byte k";

PQ_PARAM_SET  *P;
uint16_t N;
int_q q;
int_p p;
uint16_t d1;
uint16_t d2;
uint16_t d3;
int_q norm_bound_s;
int_q norm_bound_t;

int dd;
int sizeRq;
int sizeRp;
int sizeRppp;

size_t scratch_len;
unsigned char *scratch;
size_t offset;
int_q		*h_h;
uint16_t	*h_F;
uint16_t	*h_G;
int_p		*h_sp;
int_p		*h_tp;
int_p		*h_ginv;

int_q		*d_h;
//uint16_t	*d_F;
//uint16_t	*d_G;
int_p		*d_sp;
int_p		*d_tp;
int_p		*d_ginv;

unsigned	*h_rng;
unsigned	*d_rng;

int_q	*d_sig;
int 	*d_pos;
int 	*d_numValid;
int 	*numValid;
int 	*pos;
int_q	*res;

int numBlocks;

void cuda_prep(PQ_PARAM_SET_ID id) {
	int deviceID = 0;
	CSC(cudaSetDevice(deviceID));
	CSC(cudaDeviceReset());

	//// parameters
	P = pq_get_param_set_by_id(id);
	N = P->N;
	q = (int_q)P->q;
	p = (int_p)P->p;
	d1 = P->d1;
	d2 = P->d2;
	d3 = P->d3;
	norm_bound_s = P->norm_bound_s;
	norm_bound_t = P->norm_bound_t;
	dd = 2*(d1+d2+d3);
	sizeRq = N*sizeof(int_q);
	sizeRp = N*sizeof(int_p);
	sizeRppp = dd*sizeof(uint16_t);

	//// input: host/device/texture memory
	scratch_len = sizeRq+2*sizeRppp+3*sizeRp;
	CSC(cudaMallocHost(&scratch, scratch_len));
	offset = 0;
	h_h	= (int_q *)(&scratch[offset]);		offset += sizeRq;
	h_F	= (uint16_t *)(&scratch[offset]);	offset += sizeRppp;
	h_G	= (uint16_t *)(&scratch[offset]);	offset += sizeRppp;
	h_sp	= (int_p *)(&scratch[offset]);		offset += sizeRp;
	h_tp	= (int_p *)(&scratch[offset]);		offset += sizeRp;
	h_ginv = (int_p *)(&scratch[offset]);		offset += sizeRp;

	CSC(cudaMalloc(&d_h, sizeRq));
//	CSC(cudaMalloc(&d_F, sizeRppp));
//	CSC(cudaMalloc(&d_G, sizeRppp));
	CSC(cudaMalloc(&d_sp, sizeRp));
	CSC(cudaMalloc(&d_tp, sizeRp));
	CSC(cudaMalloc(&d_ginv, sizeRp));

	CSC(cudaMallocHost(&h_rng, 16*sizeof(unsigned)));
	CSC(cudaMalloc(&d_rng, 16*sizeof(unsigned)));

	CSC(cudaBindTexture(NULL, tex_sp, d_sp, sizeRp));
	CSC(cudaBindTexture(NULL, tex_tp, d_tp, sizeRp));
	CSC(cudaBindTexture(NULL, tex_h, d_h, sizeRq));
	CSC(cudaBindTexture(NULL, tex_ginv, d_ginv, sizeRp));
//	CSC(cudaBindTexture(NULL, tex_F, d_F, sizeRppp));
//	CSC(cudaBindTexture(NULL, tex_G, d_G, sizeRppp));
	CSC(cudaBindTexture(NULL, tex_rng, d_rng, 16*sizeof(unsigned)));

	memcpy(h_rng, sigma, 4);
	memcpy(h_rng+5, sigma+4, 4);
	memcpy(h_rng+10, sigma+8, 4);
	memcpy(h_rng+15, sigma+12, 4);

	//// set num of blocks
	numBlocks = P->numBlocks;

	//// output: pinned/unified memory
	CSC(cudaMalloc(&d_sig, numBlocks*16*(0x1<<P->N_bits)/16*sizeof(int_q)));
	CSC(cudaMalloc(&d_pos, (numBlocks+31)/32*sizeof(int)));
	CSC(cudaMalloc(&d_numValid, sizeof(int)));
	CSC(cudaMallocHost(&numValid, sizeof(int)));
	CSC(cudaMallocHost(&pos, (numBlocks+31)/32*sizeof(int)));

	CSC(cudaMallocHost(&res, sizeRq));
}

void cuda_clean() {
	CSC(cudaFreeHost(scratch));
	CSC(cudaFreeHost(h_rng));
	CSC(cudaFree(d_h));
//	CSC(cudaFree(d_F));
//	CSC(cudaFree(d_G));
	CSC(cudaFree(d_sp));
	CSC(cudaFree(d_tp));
	CSC(cudaFree(d_ginv));
	CSC(cudaFree(d_rng));
	CSC(cudaFree(d_sig));
	CSC(cudaFree(d_pos));
	CSC(cudaFreeHost(pos));
	CSC(cudaFree(d_numValid));
	CSC(cudaFreeHost(numValid));
	CSC(cudaFreeHost(res));
}

/** CUDA pqntrusign */
int cuda_pq_sign(
		size_t					*packed_sig_len,
		unsigned char			*packed_sig,
		const size_t			private_key_len,
		const unsigned char	*private_key_blob,
		const size_t			public_key_len,
		const unsigned char	*public_key_blob,
		const size_t			msg_len,
		const unsigned char	*msg
	) {

	int deviceID = 0;
	CSC(cudaSetDevice(deviceID));
//	float et;
//	cudaEvent_t start, stop;
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);

		//// Unpack the keys
	int rc = PQNTRU_OK;
	int64_t *unpack_temp = (int64_t *)malloc(N*sizeof(int64_t));
	rc = unpack_private_key(P, h_F, h_G, unpack_temp, private_key_len, private_key_blob);
	if (PQNTRU_ERROR == rc) {
		shred(scratch, scratch_len);
		shred(unpack_temp, N*sizeof(int64_t));
		CSC(cudaFree(scratch));
		free(unpack_temp);
		return PQNTRU_ERROR;
	}
	for (int i=0; i<N; i++)
		h_ginv[i] = (int_p)unpack_temp[i];
	rc = unpack_public_key(P, unpack_temp, public_key_len, public_key_blob);
	if (PQNTRU_ERROR == rc) {
		shred(scratch, scratch_len);
		shred(unpack_temp, N*sizeof(int64_t));
		CSC(cudaFree(scratch));
		free(unpack_temp);
		return PQNTRU_ERROR;
	}
	for (int i=0; i<N; i++)
		h_h[i] = (int_q)unpack_temp[i];

	//// Generate a document hash to sign
	challenge(h_sp, h_tp, public_key_len, public_key_blob, msg_len, msg);

	//// Copy input to GPU
	CSC(cudaMemcpy(d_sp, h_sp, sizeRp, cudaMemcpyHostToDevice));
	CSC(cudaMemcpy(d_tp, h_tp, sizeRp, cudaMemcpyHostToDevice));
	CSC(cudaMemcpy(d_h, h_h, sizeRq, cudaMemcpyHostToDevice));
	CSC(cudaMemcpy(d_ginv, h_ginv, sizeRp, cudaMemcpyHostToDevice));
//	CSC(cudaMemcpy(d_F, h_F, sizeRppp, cudaMemcpyHostToDevice));
//	CSC(cudaMemcpy(d_G, h_G, sizeRppp, cudaMemcpyHostToDevice));
	CSC(cudaMemcpyToSymbol(con_F, h_F, sizeRppp));
	CSC(cudaMemcpyToSymbol(con_G, h_G, sizeRppp));

//	cudaEventRecord(start);
	//// kernel setup
//	dim3 threadsPerBlock(16, (0x1<<P->N_bits)/16);
	dim3 threadsPerBlock(16, (P->N+15)/16);
	unsigned long sizeShared = 4*sizeRq+5*sizeRp+2*sizeRppp+sizeof(int)+
				threadsPerBlock.x*threadsPerBlock.y*sizeof(unsigned);

//	cudaEventRecord(stop);
//	cudaEventSynchronize(stop);
//	cudaEventElapsedTime(&et, start, stop);
//	t_prep += (double)et;


	CSC(cudaMemset(d_pos, 0, (numBlocks+31)/32*sizeof(int)));
	CSC(cudaMemset(d_numValid, 0, sizeof(int)));
	//// generate signatures
	unsigned long counter = 0;
	do {
//		cudaEventRecord(start);

		randombytes((unsigned char *)&h_rng[1], salsa20_key_bytes/2);
		randombytes((unsigned char *)&h_rng[11], salsa20_key_bytes/2);
		*(unsigned long *)&h_rng[8] = 0;
		*(unsigned long *)&h_rng[6] = counter;
		CSC(cudaMemcpy(d_rng, h_rng, 16*sizeof(unsigned), cudaMemcpyHostToDevice));

		cuda_sign<<<numBlocks, threadsPerBlock, sizeShared, 0>>>
					(d_sig, d_pos, d_numValid, N, q, p, d1, d2, d3, norm_bound_s, norm_bound_t, d_tp);

		CSC(cudaMemcpy(numValid, d_numValid, sizeof(int), cudaMemcpyDeviceToHost));
		counter += numBlocks*threadsPerBlock.y;
//		attempts += numBlocks;
//
//		cudaEventRecord(stop);
//		cudaEventSynchronize(stop);
//		cudaEventElapsedTime(&et, start, stop);
//		t_dowhile += (double)et;
	} while (numValid[0] == 0);

//	cudaEventRecord(start);
	CSC(cudaMemcpy(pos, d_pos, (numBlocks+31)/32*sizeof(int), cudaMemcpyDeviceToHost));
	int posIdx = 0;
	while (posIdx < numBlocks) {
		if (pos[posIdx/32] != 0) {
			int i = 0;
			while (i < 32) {
				if ((pos[posIdx/32]&(0x1<<i)) != 0) {
					posIdx += i;
					break;
				}
				else
					i ++;
			}
			break;
		}
		else
			posIdx += 32;
	}

	//// retrieve result from GPU
	CSC(cudaMemcpy(res, d_sig+posIdx*threadsPerBlock.x*threadsPerBlock.y, sizeRq, cudaMemcpyDeviceToHost));
	for (int i=0; i<N; i++) {
		unpack_temp[i] = (int64_t)res[i];
	}
	pack_signature(P, unpack_temp, *packed_sig_len, packed_sig);

	//// free memory
	shred(scratch, scratch_len);
	shred(unpack_temp, N*sizeof(int64_t));
	free(unpack_temp);
//	cudaEventRecord(stop);
//	cudaEventSynchronize(stop);
//	cudaEventElapsedTime(&et, start, stop);
//	t_writeback += (double)et;
	return PQNTRU_OK;
}

}
