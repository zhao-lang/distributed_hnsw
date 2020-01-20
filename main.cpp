#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "hnsw/hnsw.h"
#include "hnsw/node.h"
#include "hnsw/layer.h"

typedef float sim_t;
typedef std::string id_type;
typedef std::vector<float> data_t;
typedef hnsw::BasicNode<id_type,data_t> node_t;
typedef hnsw::BasicLayer<id_type,data_t,node_t> layer_t;

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

    hnsw::Index<sim_t,id_type,data_t,layer_t,node_t> index (mfunc, 5, 10);

    std::cout << "Number of nodes: " << index.getNodeCount() << std::endl;
    std::cout << "Number of layers: " << index.getLayerCount() << std::endl;

    data_t data (5);
    id_type id ("test");

    index.addNode(id, data);

    std::cout << "Number of nodes: " << index.getNodeCount() << std::endl;

    return 0;
}
