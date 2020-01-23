#pragma once

#include <mutex>
#include <unordered_map>
#include <vector>

namespace hnsw {

    template<typename data_t>
    class NodeInterface {
    public:

        virtual void setData(size_t id, data_t in_data) = 0;
        
    };

    template<typename data_t, typename sim_t>
    class BasicNode : public NodeInterface<data_t> {
    public:

        BasicNode() {}

        BasicNode(size_t id, data_t data) : id(id), data(data) {

        }

        ~BasicNode() {}

        void setData(size_t id, data_t in_data) {}

        inline data_t& getData() {
            return data;
        }

        inline size_t getID() {
            return id;
        }

        inline void lock() {
            guard_.lock();
        }

        inline void unlock() {
            guard_.unlock();
        }

        using distpair_t = std::pair<sim_t,BasicNode<data_t,sim_t>*>;
        std::vector<std::unordered_map<size_t,distpair_t>> neighbors;

    private:

        std::mutex guard_;

        size_t id;
        data_t data;

    };

}