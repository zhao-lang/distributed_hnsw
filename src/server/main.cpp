#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

#include "pb/distributed_hnsw.grpc.pb.h"
#include "core/core.h"
#include "cluster/cluster.h"

using grpc::Status;
using grpc::ServerContext;
using grpc::ServerBuilder;
using grpc::Server;


class DistributedHNSWServiceImpl final : public DistributedHNSW::Service, public server::HNSWCore, public server::Cluster {
    Status Join(ServerContext* context, const JoinRequest* request, JoinResponse* response) override {
        return Status::OK;
    }

    Status AddNode(ServerContext* context, const Node* request, Empty* response) override {
        return Status::OK;
    }

    Status DeleteNode(ServerContext* context, const Node* request, Empty* response) override {
        return Status::OK;
    }

    Status SearchKNN(ServerContext* context, const Node* request, SearchResult* response) override {
        return Status::OK;
    }
};

void RunServer() {
    std::string server_address("0.0.0.0:50051");
    DistributedHNSWServiceImpl service;

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;

    server->Wait();
}

int main() {
    std::cout << "Starting Distributed HNSW Server" << std::endl;

    RunServer();

    return 0;
}