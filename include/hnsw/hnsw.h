#pragma once

#include <math.h>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

#include "layer.h"
#include "node.h"


namespace hnsw {

    // metric function signature of (a, b, num_elements)
    template<typename sim_t,
             typename data_t>
    using METRICFUNC = sim_t(*)(data_t&, data_t&, size_t);
    

    template<typename sim_t,
             typename id_type,
             typename data_t,
             typename layer_t,
             typename node_t>
    class Index {
        struct CompareByFirst {
            constexpr bool operator()(std::pair<sim_t, node_t> const &a,
                                        std::pair<sim_t, node_t> const &b) const noexcept {
                return a.first < b.first;
            }
        };
        using queue_t = std::priority_queue<std::pair<sim_t, node_t>, std::vector<std::pair<sim_t, node_t>>, CompareByFirst>;
        
    public:

        Index() {

        }

        Index(METRICFUNC<sim_t,data_t> mfunc, size_t M = 16, size_t ef_construction = 200) {

            M_ = M;
            Mmax_ = M * 2;
            ef_construction_ = std::max(ef_construction, M_);

            levelMult_ = 1 / log(1.0 * M_);

            node_count_ = 0;

            std::unique_ptr<layer_t> l(new layer_t());
            layers.push_back(std::move(l));
        }

        ~Index() {}

        void addNode(id_type& id, data_t& data) {

            if (node_count_ == 0) {
                std::unique_ptr<node_t> node_ptr(new node_t(id, data));
                layers_guard_.lock();
                layers[0]->addNode(std::move(node_ptr));
                layers_guard_.unlock();

                node_count_guard_.lock();
                node_count_++;
                node_count_guard_.unlock();
            }
        }

        queue_t searchKnn(data_t, size_t k) {
            
        }

        size_t getNodeCount() {
            return node_count_;
        }

        size_t getLayerCount() {
            return layers.size();
        }

    private:

        
        size_t M_;                                      // out vertexes per node
        size_t Mmax_;                                   // max number of out vertexes per node
        size_t ef_construction_;                        // size of dynamic candidate list

        double levelMult_;                              // level generation factor

        std::mutex node_count_guard_;                   // mutex for node_count_
        size_t node_count_;                             // current number of nodes in graph
        
        std::mutex layers_guard_;                       // mutex for layers
        std::vector<std::unique_ptr<layer_t>> layers;   // vector of smart pointer to layers

        queue_t candidate_queue;

        void insert(data_t data, id_type id, size_t M, size_t Mmax, size_t efConstruction, size_t factor) {

        }

        queue_t search_level(node_t query, size_t level, size_t factor) {

        }

        void connect_neighbors(node_t x, size_t M, size_t level) {

        }

    };

}