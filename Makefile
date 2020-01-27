PROTOC = protoc
GRPC_CPP_PLUGIN = grpc_cpp_plugin
GRPC_CPP_PLUGIN_PATH ?= `which $(GRPC_CPP_PLUGIN)`

PROTOS_PATH = ./protos
OUT_PATH = ./include/pb


all: grpc.pb.cc pb.cc

grpc.pb.cc:
	$(PROTOC) -I $(PROTOS_PATH) --grpc_out=$(OUT_PATH) --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN_PATH) $(PROTOS_PATH)/distributed_hnsw.proto

pb.cc:
	$(PROTOC) -I $(PROTOS_PATH) --cpp_out=$(OUT_PATH) $(PROTOS_PATH)/distributed_hnsw.proto

