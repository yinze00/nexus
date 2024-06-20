#pragma once

#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/shape_inference.h"
// #include <tensorflow/core/lib/core/blocking_counter.h>
#include "nexus/turing/common/query_resource.hh"
#include "nexus/turing/common/session_resource.hh"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace nexus {
namespace turing {

#define GET_SESSION_RESOURCE(ctx)                                            \
    ({                                                                       \
        auto device = dynamic_cast<tensorflow::LocalDevice*>(ctx->device()); \
        OP_REQUIRES(ctx, device,                                             \
                    ::tensorflow::Status(tensorflow::error::UNAVAILABLE,     \
                                         "not a local device"));             \
        auto session_resource = device->GetSessionResource().get();          \
        OP_REQUIRES(ctx, session_resource,                                   \
                    ::tensorflow::Status(tensorflow::error::UNAVAILABLE,     \
                                         "SessionResource is null"));        \
        session_resource;                                                    \
    })

#define GET_SESSION_RESOURCE_PTR(ctx)                                        \
    ({                                                                       \
        auto device = dynamic_cast<tensorflow::LocalDevice*>(ctx->device()); \
        OP_REQUIRES(ctx, device,                                             \
                    ::tensorflow::Status(tensorflow::error::UNAVAILABLE,     \
                                         "not a local device"));             \
        auto session_resource = device->GetSessionResource();                \
        OP_REQUIRES(ctx, session_resource,                                   \
                    ::tensorflow::Status(tensorflow::error::UNAVAILABLE,     \
                                         "SessionResource is null"));        \
        session_resource;                                                    \
    })

#define GET_QUERY_RESOURCE(session_resource)                                  \
    ({                                                                        \
        int64_t run_id = ctx->step_id();                                      \
        auto query_resource =                                                 \
            session_resource->getQueryResource(run_id).get();                 \
        OP_REQUIRES(ctx, query_resource,                                      \
                    errors::Unavailable("invalid query resource: ", run_id)); \
        query_resource;                                                       \
    })

#define GET_QUERY_RESOURCE_PTR(session_resource)                              \
    ({                                                                        \
        int64_t run_id = ctx->step_id();                                      \
        auto query_resource = session_resource->getQueryResource(run_id);     \
        OP_REQUIRES(ctx, query_resource,                                      \
                    errors::Unavailable("invalid query resource: ", run_id)); \
        query_resource;                                                       \
    })

}  // namespace turing
}  // namespace nexus