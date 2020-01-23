#pragma once

#include <math.h>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <unordered_map>
#include <vector>

#include "node.h"


namespace hnsw {

    // Result type
    template<typename data_t>
    struct Result {
        size_t id;
        data_t& data;

        Result(size_t id, data_t& data) : id(id), data(data) {}
    };

    // metric function signature of (a, b, num_elements)
    template<typename sim_t,
             typename data_t>
    using METRICFUNC = sim_t(*)(data_t&, data_t&, size_t);
    

    template<typename sim_t,
             typename data_t,
             typename node_t>
    class Index {
        using distpair_t = std::pair<sim_t,node_t*>;

        struct CompareByFirstLess {
            constexpr bool operator()(distpair_t const &a,
                                      distpair_t const &b) const noexcept {
                return a.first < b.first;
            }
        };

        struct CompareByFirstGreater {
            constexpr bool operator()(distpair_t const &a,
                                      distpair_t const &b) const noexcept {
                return a.first > b.first;
            }
        };
        
        using queue_desc_t = std::priority_queue<distpair_t, std::vector<distpair_t>, CompareByFirstLess>;
        using queue_asc_t = std::priority_queue<distpair_t, std::vector<distpair_t>, CompareByFirstGreater>;

        using resultpair_t = std::pair<sim_t,Result<data_t>>;
        using nodesmap_t = std::unordered_map<size_t,std::unique_ptr<node_t>>;
        

    public:

        Index() {

        }

        Index(METRICFUNC<sim_t,data_t> mfunc, size_t data_dim, size_t M = 16, size_t ef_construction = 200, size_t seed = 12345) {
            
            mfunc_ = mfunc;
            data_dim_ = data_dim;

            M_ = M;
            Mmax_ = M_;
            Mmax0_ = M_ * 2;
            ef_construction_ = std::max(ef_construction, M_);

            level_mult_ = 1 / log(1.0 * M_);

            node_count_ = 0;
            max_layer_ = 0;

            rng_.seed(seed);
        }

        ~Index() {}

        void addNode(size_t id, data_t data) {
            if (node_count_ == 0) {
                nodes_guard_.lock();
                nodes[id] = std::unique_ptr<node_t> (new node_t(id, data));
                nodes_guard_.unlock();

                node_count_guard_.lock();
                node_count_++;
                node_count_guard_.unlock();

                enterpoint_guard_.lock();
                enterpoint = nodes[id].get();
                enterpoint_guard_.unlock();

                return;
            }

            if (nodes.find(id) != nodes.end()) {
                return;
            }

            insert(id, data, M_, Mmax_, ef_construction_, level_mult_);
        }

        void deleteNode(size_t id) {
            if (nodes.find(id) == nodes.end()) {
                return;
            } else {
                for (size_t lc = 0; lc < std::min(max_layer_+1, nodes[id]->neighbors.size()); lc++) {
                    deleteNodeNeighbors(nodes[id].get(), Mmax_, lc);
                }

                nodes_guard_.lock();
                nodes.erase(id);
                nodes_guard_.unlock();

                node_count_guard_.lock();
                node_count_--;
                node_count_guard_.unlock();
            }
        }

        std::vector<resultpair_t> searchKnn(data_t& data, size_t K) {
            std::vector<resultpair_t> res;

            if (node_count_ == 0 || !enterpoint) {
                return res;
            }

            res = searchKnnInternal(data, K, ef_construction_);

            return res;
        }

        size_t getNodeCount() {
            return node_count_;
        }

        size_t getLayerCount() {
            return max_layer_ + 1;
        }
        
    private:

        METRICFUNC<sim_t,data_t> mfunc_;                // similarity metric function
        size_t data_dim_;                               // dimensionality of the data
        
        size_t M_;                                      // out vertexes per node
        size_t Mmax_;                                   // max number of out vertexes per node
        size_t Mmax0_;                                  // max number of out vertexes per node at layer 0
        size_t ef_construction_;                        // size of dynamic candidate list

        double level_mult_;                             // level generation factor

        std::mutex node_count_guard_;                   // mutex for node_count_
        size_t node_count_;                             // current number of nodes in graph

        std::mutex max_layer_guard_;                    // mutex for v
        size_t max_layer_;                              // index of top layer
        
        std::mutex nodes_guard_;                        // mutex for nodes
        nodesmap_t nodes;                              // vector of smart pointers to nodes

        std::mutex enterpoint_guard_;
        node_t *enterpoint;

        std::default_random_engine rng_;

        void insert(size_t id, data_t& data, size_t M, size_t Mmax, size_t ef_construction, double level_mult) {
            
            queue_desc_t W;
            node_t *ep = enterpoint;
            size_t L = max_layer_;
            size_t l = genRandomLevel(level_mult);

            nodes_guard_.lock();
            nodes[id] = std::unique_ptr<node_t> (new node_t(id, data));
            nodes_guard_.unlock();

            node_count_guard_.lock();
            node_count_++;
            node_count_guard_.unlock();

            size_t lc = L;
            while (lc >= l+1) {
                W = search_level(data, ep, 1, lc);
                while (W.size() > 1) {
                    W.pop();
                }
                ep = W.top().second;
                
                if (lc == 0) {
                    break;
                }
                lc--;
            }

            lc = std::min(L, l);
            while (lc >= 0) {
                W = search_level(data, ep, ef_construction, lc);
                queue_desc_t neighbors = select_neighbors(nodes[id].get(), W, M, lc, true, true);
                connect_neighbors(nodes[id].get(), queue_desc_t(neighbors), lc);

                // shrink connections as needed
                while (neighbors.size() > 0) {
                    distpair_t e = neighbors.top();
                    neighbors.pop();

                    std::unordered_map<size_t,distpair_t> eConnMap = e.second->neighbors[lc];
                    queue_desc_t eConn;
                    for (auto & n : eConnMap) {
                        eConn.push(n.second);
                    }

                    size_t _Mmax = (lc == 0) ? Mmax0_ : Mmax;
                    if (eConn.size() > _Mmax) {
                        queue_desc_t eNewConn = select_neighbors(e.second, eConn, _Mmax, lc, true, true);

                        e.second->lock();
                        e.second->neighbors[lc].clear();
                        while (eNewConn.size() > 0) {
                            distpair_t newpair = eNewConn.top();
                            eNewConn.pop();
                            e.second->neighbors[lc][newpair.second->getID()] = newpair;
                        }
                        e.second->unlock();
                    }
                }

                ep = W.top().second;

                if (lc == 0) {
                    break;
                }
                lc--;
            }

            if (l > L) {
                max_layer_guard_.lock();
                max_layer_ = l;
                max_layer_guard_.unlock();

                // set enterpoint
                enterpoint_guard_.lock();
                enterpoint = nodes[id].get();
                enterpoint_guard_.unlock();
            }
        }

        size_t genRandomLevel(double level_mult) {
            std::uniform_real_distribution<double> dist (0.0, 1.0);
            size_t level = (size_t) -log(dist(rng_)) * level_mult;
            return level;
        }

        queue_desc_t search_level(data_t& query, node_t* ep, size_t ef, size_t level) {
            std::unordered_map<size_t, bool> v;
            queue_desc_t C;
            queue_asc_t W;

            sim_t qsim = mfunc_(query, ep->getData(), data_dim_);
            distpair_t qpair (qsim, ep);
            v[ep->getID()] = true; 
            C.push(qpair);
            W.push(qpair);

            while (C.size() > 0) {
                distpair_t cpair = C.top();
                C.pop();

                distpair_t fpair = W.top();

                if (cpair.first < fpair.first) {
                    break;
                }

                // update C and W
                cpair.second->lock();
                while (cpair.second->neighbors.size() < level+1) {
                    cpair.second->neighbors.push_back(std::unordered_map<size_t,distpair_t>());
                }
                cpair.second->unlock();
                for (auto & emap : cpair.second->neighbors[level]) {
                    distpair_t e = emap.second;
                    size_t eid = e.second->getID();
                    if (v.find(eid) == v.end() || !v.find(eid)->second) {
                        v[eid] = true;
                        
                        fpair = W.top();
                        sim_t esim = mfunc_(query, e.second->getData(), data_dim_);
                        if (esim > fpair.first || W.size() < ef) {
                            distpair_t epair (esim, e.second);
                            C.push(epair);
                            W.push(epair);

                            if (W.size() > ef) {
                                W.pop();
                            }
                        }
                    }
                }
            }

            queue_desc_t res;
            while (W.size() > 0) {
                res.push(W.top());
                W.pop();
            }

            return res;
        }

        queue_desc_t select_neighbors(node_t* query, queue_desc_t& C, size_t M, size_t lc, bool extendCandidates, bool keepPrunedConnections, node_t* ignoredNode = nullptr) {
            queue_desc_t R;
            queue_desc_t W(C);
            queue_desc_t Wd;
            
            // extend candidates by their neighbors
            if (extendCandidates) {
                queue_desc_t Ccopy(C);

                std::unordered_map<size_t,bool> wmap;
                while (Ccopy.size() > 0) {
                    distpair_t epair = Ccopy.top();
                    Ccopy.pop();
                    wmap[epair.second->getID()] = true;
                }

                Ccopy = queue_desc_t(C);
                while (Ccopy.size() > 0) {
                    distpair_t epair = Ccopy.top();
                    Ccopy.pop();

                    if (epair.second == ignoredNode) {
                        continue;
                    }

                    for (auto & eadjmap : epair.second->neighbors[lc]) {
                        if (eadjmap.second.second == ignoredNode || eadjmap.second.second == query) {
                            continue;
                        }
                        if (wmap.find(eadjmap.first) == wmap.end() || !wmap.find(eadjmap.first)->second) {
                            sim_t eadj_sim = mfunc_(query->getData(), eadjmap.second.second->getData(), data_dim_);
                            distpair_t eadjpair (eadj_sim, query);
                            W.push(eadjpair);
                        }
                    }
                }
            }

            while (W.size() > 0 && R.size() < M) {
                distpair_t epair = W.top();
                W.pop();

                if (epair.second == ignoredNode || epair.second == query) {
                    continue;
                }

                if (R.size() == 0 || epair.first > R.top().first) {
                    R.push(epair);
                } else {
                    Wd.push(epair);
                }
            }

            if (keepPrunedConnections) { // add back some of the discarded connections
                while (Wd.size() > 0 && R.size() < M) {
                    distpair_t ppair = Wd.top();
                    Wd.pop();

                    if (ppair.second == ignoredNode || ppair.second == query) {
                        continue;
                    }
                    R.push(ppair);
                }
            }

            return R;
        }

        void connect_neighbors(node_t* q, queue_desc_t neighbors, size_t level) {
            q->lock();

            while (q->neighbors.size() < level+1) {
                q->neighbors.push_back(std::unordered_map<size_t,distpair_t>());
            }

            while (neighbors.size() > 0) {
                distpair_t npair = neighbors.top();
                neighbors.pop();

                npair.second->lock();

                while (npair.second->neighbors.size() < level+1) {
                    npair.second->neighbors.push_back(std::unordered_map<size_t,distpair_t>());
                }

                q->neighbors[level][npair.second->getID()] = npair;
                npair.second->neighbors[level][q->getID()] = distpair_t(npair.first,q);

                npair.second->unlock();
            }

            q->unlock();
        }

        void deleteNodeNeighbors(node_t* node, size_t Mmax, size_t lc) {
            auto & neighborhood = node->neighbors[lc];

            for (auto & emap : neighborhood) {
                // select new neighbors while excluding node
                distpair_t e = emap.second;
                
                std::unordered_map<size_t,distpair_t> eConnMap = e.second->neighbors[lc];
                queue_desc_t eConn;
                for (auto & n : eConnMap) {
                    eConn.push(n.second);
                }

                // TODO this is not reconstructing the new connections correctly
                size_t _Mmax = (lc == 0) ? Mmax0_ : Mmax;
                queue_desc_t eNewConn = select_neighbors(e.second, eConn, _Mmax, lc, true, true, node);

                e.second->lock();
                e.second->neighbors[lc].clear();
                while (eNewConn.size() > 0) {
                    distpair_t newpair = eNewConn.top();
                    eNewConn.pop();
                    e.second->neighbors[lc][newpair.second->getID()] = newpair;
                }
                e.second->unlock();
            }
        }

        std::vector<resultpair_t> searchKnnInternal(data_t& query, size_t K, size_t ef) {
            queue_desc_t W;
            node_t* ep = enterpoint;
            size_t L = max_layer_;

            size_t lc = L;
            while (lc >= 1) {
                W = search_level(query, ep, 1, lc);
                ep = W.top().second;
                lc--;
            }

            W = search_level(query, ep, ef, 0);

            std::vector<resultpair_t> res;
            while (res.size() < K && W.size() > 0) {
                distpair_t c = W.top();
                W.pop();

                res.push_back(
                    resultpair_t (c.first, Result(c.second->getID(), c.second->getData()))
                );   
            }

            return res;
        }

        // #include <iostream>
        // template<typename K, typename V>
        // void print_map(std::unordered_map<K,V> const &m)
        // {
        //     for (auto const& pair: m) {
        //         std::cout << "{" << pair.first << ": " << pair.second.first << "," << pair.second.second << "}\n";
        //     }
        // }

    };

}
