#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("RequestInitOp")
    .Input("topk: uint32")
    .Output("entry_point: uint32")
    .Attr("index_name: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->MakeShape({1}));
        return Status::OK();
    });

/*
 * OP: GatherNeighborsOp
 * Input:
 *   1. entry_points
 *   2. hints (must traverse the neighbors of )
 * Output:
 *   1. neighbors's inner_id
 */
REGISTER_OP("GatherNeighborsOp")
    .Input("entry_points: uint32")
    .Output("neighbors: uint32")
    .Attr("level: int")
    .Attr("nneis: int")
    .Attr("index_name: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle entry_points_shape;
        int                          nneis;
        TF_RETURN_IF_ERROR(c->GetAttr("nneis", &nneis));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &entry_points_shape));
        // // 设置 neighbors 的形状为 {entry_points * number_of_neighbors}
        // shape_inference::ShapeHandle neighbors_shape =
        //     c->MakeShape({c->Dim(entry_points_shape, 0) * nneis});

        c->set_output(0, entry_points_shape);

        return Status::OK();
    });

/*
 * OP: GatherNeighborsWithoutHintOp
 * Input:
 *   1. entry_points
 *   2. hints (must traverse the neighbors of )
 * Output:
 *   1. neighbors's inner_id
 */
REGISTER_OP("GatherNeighborsWithHintOp")
    .Input("entry_points: uint32")
    .Input("hints: uint32")
    .Output("neighbors: uint32")
    .Attr("level: int")
    .Attr("index_name: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        // shape_inference::ShapeHandle entry_points_shape;
        // TF_RETURN_IF_ERROR(
        //     c->WithRank(c->input(0), 1,
        //                 &entry_points_shape));  // 假设 entry_points
        //                 是一维张量
        // // 设置 neighbors 的形状为 {entry_points, number_of_neighbors}
        // shape_inference::ShapeHandle neighbors_shape = c->MakeShape({
        //     c->Dim(entry_points_shape, 0) * 128  // entry_points 的第一个维度
        // });

        // c->set_output(0, neighbors_shape);

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
    .Input("internal_ids: uint32")
    .Output("embeddings: float")
    .Attr("index_name: string")
    .Attr("dim: int")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        // 获取 internal_ids 的形状
        shape_inference::ShapeHandle internal_ids_shape;
        TF_RETURN_IF_ERROR(
            c->WithRank(c->input(0), 1,
                        &internal_ids_shape));  // 假设 internal_ids 是一维张量

        // 获取 dim 属性的值
        int64 dim;
        TF_RETURN_IF_ERROR(c->GetAttr("dim", &dim));

        // 设置 embeddings 的形状为 {internal_ids.size(), dim}
        shape_inference::ShapeHandle embeddings_shape = c->MakeShape({
            c->Dim(internal_ids_shape, 0),  // internal_ids 的第一个维度
            dim                             // 固定的 dim
        });

        // 设置 output 形状
        c->set_output(0, embeddings_shape);
        return Status::OK();
    });

/*
 * OP: GEMV inner product
 * Input:
 *   1. inner_ids
 * Output:
 *   1. inner_ids's embeddings
 */
REGISTER_OP("GemvOp")
    .Input("user_emb: float")
    .Input("item_emb: float")
    .Output("ug_similarity: float")
    .Attr("T: {float, double} = DT_FLOAT")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle matrix_shape, vector_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &vector_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &matrix_shape));

        // 确定输出形状为 [batch_size, num_cols]
        shape_inference::DimensionHandle batch_size = c->Dim(matrix_shape, 0);
        // shape_inference::DimensionHandle num_cols   = c->Dim(matrix_shape,
        // 1);
        c->set_output(0, c->MakeShape({batch_size}));

        return Status::OK();
    });

REGISTER_OP("IndirectSortAndTopkOp")
    .Input("k: T")
    .Input("v: U")
    .Output("sorted_k: T")
    .Output("sorted_v: U")
    .Attr("T: {uint32, uint64} = DT_UINT32")
    .Attr("U: {float, double} = DT_FLOAT")
    .Attr("topk: int >= 0 = 1000")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle input1_shape, input2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input1_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &input2_shape));
        c->set_output(0, input1_shape);
        c->set_output(1, input2_shape);
        return Status::OK();
    });

REGISTER_OP("ResultConstructOp")
    .Input("k: uint32")
    .Input("v: float")
    .Output("results_labels: uint32")
    .Output("results_scores: float")
    .Attr("index_name: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_shape));
        c->set_output(0, input_shape);
        c->set_output(1, input_shape);
        return Status::OK();
    });

REGISTER_OP("MLPOp")
    .Input("user_emb: float")
    .Input("item_emb: float")
    .Output("ug_similarity: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        return Status::OK();
    });
