#pragma once


namespace hnsw {

    template<typename key_t,
             typename data_t>
    class NodeInterface {
    public:

        virtual void setData(key_t key, const void *in_data) = 0;
    
    protected:

        key_t key;
        data_t data;

    };

    template<typename key_t,
             typename data_t>
    class BasicNode : public NodeInterface<key_t,data_t> {
    public:

    private:

    };

}