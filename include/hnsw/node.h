#pragma once

#include <mutex>

namespace hnsw {

    template<typename id_type,
             typename data_t>
    class NodeInterface {
    public:

        virtual void setData(id_type key, data_t in_data) = 0;
        
    };

    template<typename id_type,
             typename data_t>
    class BasicNode : public NodeInterface<id_type,data_t> {
    public:

        BasicNode() {}

        BasicNode(id_type id, data_t data) : id(id), data(data) {

        }

        ~BasicNode() {}

        void setData(id_type key, data_t in_data) {}

    private:

        std::mutex guard_;

        id_type id;
        data_t data;

    };

}