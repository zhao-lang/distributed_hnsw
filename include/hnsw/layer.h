#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "node.h"

namespace hnsw {

    template<typename id_type,
             typename data_t,
             typename node_t>
    class LayerInterface {
    public:

        virtual void addNode(std::unique_ptr<node_t> node) = 0;
        virtual void deleteNode(std::unique_ptr<node_t> node) = 0;

    };

    template<typename id_type,
             typename data_t,
             typename node_t>
    class BasicLayer : public LayerInterface<id_type,data_t,node_t> {
    public:

        BasicLayer() {}

        ~BasicLayer() {}

        void addNode(std::unique_ptr<node_t> node_ptr) {
            guard_.lock();
            nodes.push_back(std::move(node_ptr));
            guard_.unlock();
        }

        void deleteNode(std::unique_ptr<node_t> node) {}

    private:

        std::mutex guard_;

        std::vector<std::unique_ptr<node_t>> nodes;

    };
}