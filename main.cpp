#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "hnsw/hnsw.h"
#include "hnsw/node.h"

typedef float sim_t;
typedef std::vector<float> data_t;
typedef hnsw::BasicNode<data_t,sim_t> node_t;

static float sim_func(data_t& a, data_t& b, size_t n) {
    float res = 0;
    for (unsigned i = 0; i < n; i++) {
        float t = a[i] - b[i];
        res += t * t;
    }
    return res;
}

int main() {

    hnsw::METRICFUNC<sim_t,data_t> mfunc = sim_func;
    size_t data_dim = 5;

    hnsw::Index<sim_t,data_t,node_t> index (mfunc, data_dim, 5, 10);

    std::cout << "Number of nodes: " << index.getNodeCount() << std::endl;
    std::cout << "Number of layers: " << index.getLayerCount() << std::endl;

    data_t data (data_dim, 1.0);
    size_t id = 1;

    index.addNode(id, data);

    std::cout << "Number of nodes: " << index.getNodeCount() << std::endl;

    return 0;
}
