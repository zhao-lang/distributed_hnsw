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
                std::unique_ptr<node_t> node_ptr(new node_t(id, data));

                enterpoint = node_ptr.get();

                node_count_guard_.lock();
                node_count_++;
                node_count_guard_.unlock();

                nodes_guard_.lock();
                nodes.push_back(std::move(node_ptr));
                nodes_guard_.unlock();

                return;
            }

            insert(id, data, M_, Mmax_, ef_construction_, level_mult_);
        }

        std::vector<resultpair_t> searchKnn(data_t& data, size_t K) {
            return searchKnnInternal(data, K, ef_construction_);
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
        std::vector<std::unique_ptr<node_t>> nodes;     // vector of smart pointers to nodes

        node_t *enterpoint;

        std::default_random_engine rng_;

        void insert(size_t& id, data_t& data, size_t M, size_t Mmax, size_t ef_construction, double level_mult) {
            
            queue_desc_t W;
            node_t *ep = enterpoint;
            size_t L = max_layer_;
            size_t l = genRandomLevel(level_mult);

            std::unique_ptr<node_t> node_ptr(new node_t(id, data));

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
                queue_desc_t neighbors = select_neighbors(node_ptr.get(), W, M, lc, true, true);
                connect_neighbors(node_ptr.get(), queue_desc_t(neighbors), lc);

                // shrink connections as needed
                // TODO keep sim_t in neighbors map so it doesn't have to get calced again
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
                        e.second->neighbors[lc].clear();
                        while (eNewConn.size() > 0) {
                            distpair_t newpair = eNewConn.top();
                            eNewConn.pop();
                            e.second->neighbors[lc][newpair.second->getID()] = newpair;
                        }
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
                enterpoint = node_ptr.get();
            }

            node_count_guard_.lock();
            node_count_++;
            node_count_guard_.unlock();

            nodes_guard_.lock();
            nodes.push_back(std::move(node_ptr));
            nodes_guard_.unlock();

            // for (auto & n : nodes) {
            //     std::cout << n->neighbors.size() << ", " << n->neighbors[0].size() << std::endl;
            //     print_map(n->neighbors[0]);
            // }
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
                while (cpair.second->neighbors.size() < level+1) {
                    cpair.second->neighbors.push_back(std::unordered_map<size_t,distpair_t>());
                }
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

        queue_desc_t select_neighbors(node_t* query, queue_desc_t& C, size_t M, size_t lc, bool extendCandidates, bool keepPrunedConnections) {
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
                    for (auto & eadjmap : epair.second->neighbors[lc]) {
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
                    R.push(ppair);
                }
            }

            return R;
        }

        void connect_neighbors(node_t* q, queue_desc_t neighbors, size_t level) {
            while (q->neighbors.size() < level+1) {
                q->neighbors.push_back(std::unordered_map<size_t,distpair_t>());
            }

            while (neighbors.size() > 0) {
                distpair_t npair = neighbors.top();
                neighbors.pop();

                while (npair.second->neighbors.size() < level+1) {
                    npair.second->neighbors.push_back(std::unordered_map<size_t,distpair_t>());
                }

                q->neighbors[level][npair.second->getID()] = npair;
                npair.second->neighbors[level][q->getID()] = distpair_t(npair.first,q);
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
