#include <iostream>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <immintrin.h>

#include "hnsw/hnsw.h"


typedef float sim_t;
typedef std::vector<float> data_t;
typedef std::pair<sim_t,hnsw::Result<data_t>> resultpair_t;

static float sim_func(data_t& a, data_t& b, size_t n) {
    float res = 0;
    for (unsigned i = 0; i < n; i++) {
        float t = a[i] - b[i];
        res += t * t;
    }
    return -res;
}

float hsum_ps_sse3(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(v, shuf);
    shuf        = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums        = _mm_add_ss(sums, shuf);
    return        _mm_cvtss_f32(sums);
}

float hsum256_ps_avx(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
           vlow  = _mm_add_ps(vlow, vhigh);     // add the low 128
    return hsum_ps_sse3(vlow);         // and inline the sse3 version, which is optimal for AVX
}

// Multiple accumulators and FMA
// since FMA has a latency of 5 cycles but 0.5 CPI
// https://stackoverflow.com/questions/45735679/euclidean-distance-using-intrinsic-instruction
// TODO: extend functionality for vectors of non-multiples of 32 floats
static float sim_func_avx(data_t& a, data_t& b, size_t n) {

    __m256 euc1 = _mm256_setzero_ps();
    __m256 euc2 = _mm256_setzero_ps();
    __m256 euc3 = _mm256_setzero_ps();
    __m256 euc4 = _mm256_setzero_ps();

    for (size_t i = 0; i < n; i += 8*4) {
        const __m256 v1 = _mm256_sub_ps(_mm256_loadu_ps(&a[i + 0]), _mm256_loadu_ps(&b[i + 0]));
        euc1 = _mm256_fmadd_ps(v1, v1, euc1);

        const __m256 v2 = _mm256_sub_ps(_mm256_loadu_ps(&a[i + 8]), _mm256_loadu_ps(&b[i + 8]));
        euc2 = _mm256_fmadd_ps(v2, v2, euc2);

        const __m256 v3 = _mm256_sub_ps(_mm256_loadu_ps(&a[i + 16]), _mm256_loadu_ps(&b[i + 16]));
        euc3 = _mm256_fmadd_ps(v3, v3, euc3);

        const __m256 v4 = _mm256_sub_ps(_mm256_loadu_ps(&a[i + 24]), _mm256_loadu_ps(&b[i + 24]));
        euc4 = _mm256_fmadd_ps(v4, v4, euc4);
    }

    float res = hsum256_ps_avx(_mm256_add_ps(_mm256_add_ps(euc1, euc2), _mm256_add_ps(euc3, euc4)));
    
    return -res;
} 

int main() {

    // hnsw::METRICFUNC<sim_t,data_t> mfunc = sim_func;
    hnsw::METRICFUNC<sim_t,data_t> mfunc = sim_func_avx;
    size_t data_dim = 512;

    hnsw::Index<sim_t,data_t> index (mfunc, data_dim, 5, 10);

    std::cout << "Number of nodes: " << index.getNodeCount() << std::endl;
    std::cout << "Number of layers: " << index.getLayerCount() << std::endl;

    size_t n = 10000;
    std::cout << "Adding " << n << " nodes" << std::endl;
    auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < 10000; i++) {
        data_t data (data_dim, float(i));
        size_t id = i;

        index.addNode(id, data);
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
        << "ms\n" << std::endl;

    std::cout << "Number of nodes: " << index.getNodeCount() << std::endl;

    float datum = 100.0;
    data_t query (data_dim, datum);
    start = std::chrono::steady_clock::now();
    std::vector<resultpair_t> res = index.searchKnn(query, 5);
    end = std::chrono::steady_clock::now();

    std::cout << "Query data: " << data_dim << "x" << datum << "f" << std::endl;

    for (auto & r : res) {
        std::cout << "- similarity: " << r.first << ", ID: " << r.second.id << std::endl;
    }

    std::cout << "Number of layers: " << index.getLayerCount() << std::endl;

    std::cout << "Elapsed time: "
        << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
        << "µs\n" << std::endl;

    return 0;
}
