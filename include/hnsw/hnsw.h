#pragma once

#include <memory>
#include <queue>
#include <vector>

#include "layer.h"
#include "node.h"


namespace hnsw {

    // metric function signature of (a, b)
    template<typename METRICTYPE>
    using METRICFUNC = METRICTYPE(*)(const void *, const void *);

    template<typename METRICTYPE,
             typename sim_t,
             typename key_t,
             typename data_t>
    class Index {
        using layer_t = std::unique_ptr<LayerInterface<key_t,data_t>>;
        using node_t = std::unique_ptr<NodeInterface<key_t,data_t>>;
        using queue_t = std::priority_queue<std::pair<sim_t, node_t>, std::vector<std::pair<sim_t, node_t>>, CompareByFirst>;
        
    public:

        Index(METRICFUNC<METRICTYPE> mfunc, size_t M = 16) {

        }

        ~Index() {}

        queue_t searchKnn(data_t, size_t k) {

        }

        size_t getCount() {
            return node_count_;
        }

    private:

        size_t node_count_;  // current number of nodes in graph
        size_t M_;           // out vertexes per node
    
        std::vector<layer_t> layers;

        queue_t candidate_queue;

        queue_t search_level(node_t query, size_t level, size_t factor) {

        }

        void connect_neighbors(node_t x, size_t M, size_t level) {

        }

        struct CompareByFirst {
            constexpr bool operator()(std::pair<sim_t, node_t> const &a,
                                      std::pair<sim_t, node_t> const &b) const noexcept {
                return a.first < b.first;
            }
        };

    };

}