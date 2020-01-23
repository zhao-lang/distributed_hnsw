#include "catch.hpp"

#include <vector>
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

hnsw::METRICFUNC<sim_t,data_t> mfunc = sim_func;
size_t data_dim = 5;
hnsw::Index<sim_t,data_t,node_t> hindex (mfunc, data_dim, 5, 10);


TEST_CASE("Index attributes should update correctly") {

    SECTION("New Index should have 1 layer and 0 nodes") {
        REQUIRE(hindex.getLayerCount() == 1);
        REQUIRE(hindex.getNodeCount() == 0);
    }

    SECTION("Search on empty index should return no results") {
        data_t query (data_dim, 10.0);
        std::vector<resultpair_t> res = hindex.searchKnn(query, 5);
        REQUIRE(res.size() == 0);
    }
    
    SECTION("Adding nodes should update node_count_") {
        int n = 100;
        for (int i = 0; i < 100; i++) {
            data_t data (data_dim, float(i));
            size_t id = i;
            hindex.addNode(id, data);
        }
        REQUIRE(hindex.getNodeCount() == n);
    }

    SECTION("Search should return correct results") {
        data_t query (data_dim, 10.0);
        std::vector<resultpair_t> res = hindex.searchKnn(query, 5);

        std::vector<size_t> expected_ids = {8, 9, 10, 11, 12};

        for (auto & id : expected_ids) {
            bool found = false;
            for (auto & r : res) {
                if (r.second.id == id) {
                    found = true;
                }
            }
            REQUIRE(found == true);
        }
    }

    SECTION("Deleting a node should succeed") {
        hindex.deleteNode(10);
        REQUIRE(hindex.getNodeCount() == 99);
    }

    SECTION("Search after delete should return correct results") {
        data_t query (data_dim, 10.0);
        std::vector<resultpair_t> res = hindex.searchKnn(query, 4);

        std::vector<size_t> expected_ids = {8, 9, 11, 12};

        for (auto & id : expected_ids) {
            bool found = false;
            for (auto & r : res) {
                if (r.second.id == id) {
                    found = true;
                }
            }
            REQUIRE(found == true);
        }

        bool found = false;
        for (auto & r : res) {
            if (r.second.id == 10) {
                found = true;
            }
        }
        REQUIRE(found == false);
    }
}


