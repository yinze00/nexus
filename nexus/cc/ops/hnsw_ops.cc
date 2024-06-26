#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


// REGISTER_OP("RequestInit");

/*
* OP: GatherNeighborsOp
* Input:
*   1. entry_points
*   2. hints (must traverse the neighbors of )
* Output:
*   1. neighbors's inner_id
*/
REGISTER_OP("GatherNeighborsOp")
    .Input("entry_points: int32")
    .Input("hints: int32")
    .Output("neighbors: int32")
    .Attr("level: int")
    .Attr("index_name: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        return Status::OK();
    });

/*
* OP: GatherEmbeddingsOp
* Input:
*   1. inner_ids 
* Output:
*   1. inner_ids's embeddings
*/
REGISTER_OP("GatherEmbeddingsOp")
    .Input("internal_ids: int32")
    .Output("embeddings: float")
    .Attr("index_name: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        return Status::OK();
    });

/*
* OP: GatherEmbeddingsOp
* Input:
*   1. inner_ids 
* Output:
*   1. inner_ids's embeddings
*/
REGISTER_OP("GemvOp")
    .Input("user_emb: float")
    .Input("item_emb: float")
    .Output("ug_similarity: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle matrix_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &matrix_shape));

        shape_inference::ShapeHandle vector_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &vector_shape));

        // 确定输出形状为 [batch_size, num_cols]
        shape_inference::DimensionHandle batch_size = c->Dim(matrix_shape, 0);
        shape_inference::DimensionHandle num_cols = c->Dim(matrix_shape, 1);
        c->set_output(0, c->Matrix(batch_size, 1));

        return Status::OK();
    });

REGISTER_OP("MLPOp")
    .Input("user_emb: float")
    .Input("item_emb: float")
    .Output("ug_similarity: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        return Status::OK();
    });

// REGISTER_OP("RankAndTopK")
//     .Input("")
