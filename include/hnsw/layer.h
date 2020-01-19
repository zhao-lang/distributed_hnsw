#pragma once

#include <memory>
#include <vector>

#include "node.h"

namespace hnsw {

    template<typename key_t,
             typename data_t>
    class LayerInterface {
        using node_t = std::unique_ptr<NodeInterface<key_t,data_t>>;

    public:

        virtual void addNode(node_t) = 0;
        virtual void deleteNode(node_t) = 0;

    protected:
    
        node_t nodes;

    };

    template<typename key_t,
             typename data_t>
    class BasicLayer : public LayerInterface<key_t,data_t> {
    public:

    private:

    };
}