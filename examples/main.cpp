#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <immintrin.h>

#include "hnsw/hnsw.h"


typedef float sim_t;
typedef std::vector<float> data_t;
typedef hnsw::BasicNode<data_t,sim_t> node_t;
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
    // (no wasted instructions, and all of them are the 4B minimum)
}

static float sim_func_avx(data_t& a, data_t& b, size_t n) {
    float res;
    for (size_t i = 0; i < n; i += 8) {
        __m256 vec_a = _mm256_loadu_ps(&a[i]);
        __m256 vec_b = _mm256_loadu_ps(&b[i]);
        __m256 vec_t = _mm256_sub_ps(vec_a, vec_b);
        __m256 vec_t2 = _mm256_mul_ps(vec_t, vec_t);
        res += hsum256_ps_avx(vec_t2);
    }
    return -res;
} 

int main() {

    // hnsw::METRICFUNC<sim_t,data_t> mfunc = sim_func;
    hnsw::METRICFUNC<sim_t,data_t> mfunc = sim_func_avx;
    size_t data_dim = 16;

    hnsw::Index<sim_t,data_t,node_t> index (mfunc, data_dim, 5, 10);

    std::cout << "Number of nodes: " << index.getNodeCount() << std::endl;
    std::cout << "Number of layers: " << index.getLayerCount() << std::endl;

    for (size_t i = 0; i < 100; i++) {
        data_t data (data_dim, float(i));
        size_t id = i;

        index.addNode(id, data);
    }

    std::cout << "Number of nodes: " << index.getNodeCount() << std::endl;

    data_t query (data_dim, 10.0);
    std::vector<resultpair_t> res = index.searchKnn(query, 5);

    for (auto & r : res) {
        std::cout << "- similarity: " << r.first << ", ID: " << r.second.id << std::endl;
    }

    std::cout << "Number of layers: " << index.getLayerCount() << std::endl;

    return 0;
}
