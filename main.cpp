#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "hnsw/hnsw.h"
#include "hnsw/node.h"

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

int main() {

    hnsw::METRICFUNC<sim_t,data_t> mfunc = sim_func;
    size_t data_dim = 5;

    hnsw::Index<sim_t,data_t,node_t> index (mfunc, data_dim, 5, 10);

    std::cout << "Number of nodes: " << index.getNodeCount() << std::endl;
    std::cout << "Number of layers: " << index.getLayerCount() << std::endl;

    for (size_t i = 0; i < 20; i++) {
        data_t data (data_dim, float(i));
        size_t id = i;

        index.addNode(id, data);
    }

    std::cout << "Number of nodes: " << index.getNodeCount() << std::endl;

    data_t query (data_dim, 0.0);
    std::vector<resultpair_t> res = index.searchKnn(query, 4);

    for (auto & r : res) {
        std::cout << "- similarity: " << r.first << ", ID: " << r.second.id << std::endl;
    }

    return 0;
}
