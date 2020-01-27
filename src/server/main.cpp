#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

#include "pb/distributed_hnsw.grpc.pb.h"


class DistributedHNSWServiceImpl final : public DistributedHNSW::Service {
    grpc::Status Join(grpc::ServerContext* context, const JoinRequest* request, JoinResponse* response) override {
        
    }
};

int main() {
    std::cout << "Starting Distributed HNSW Server" << std::endl;
}