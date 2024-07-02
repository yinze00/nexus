#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("RequestInitOp")
    .Output("entry_point: int32")
    .Attr("index_name: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->Scalar());
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
    .Input("entry_points: int32")
    .Output("neighbors: int32")
    .Attr("level: int")
    .Attr("index_name: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle entry_points_shape;
        TF_RETURN_IF_ERROR(
            c->WithRank(c->input(0), 1,
                        &entry_points_shape));  // 假设 entry_points 是一维张量
        // 设置 neighbors 的形状为 {entry_points, number_of_neighbors}
        shape_inference::ShapeHandle neighbors_shape = c->MakeShape({
            c->Dim(entry_points_shape, 0),  // entry_points 的第一个维度
            128                             // 固定的 number_of_neighbors
        });

        c->set_output(0, neighbors_shape);

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
    .Input("entry_points: int32")
    .Input("hints: int32")
    .Output("neighbors: int32")
    .Attr("level: int")
    .Attr("index_name: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle entry_points_shape;
        TF_RETURN_IF_ERROR(
            c->WithRank(c->input(0), 1,
                        &entry_points_shape));  // 假设 entry_points 是一维张量
        // 设置 neighbors 的形状为 {entry_points, number_of_neighbors}
        shape_inference::ShapeHandle neighbors_shape = c->MakeShape({
            c->Dim(entry_points_shape, 0),  // entry_points 的第一个维度
            128                             // 固定的 number_of_neighbors
        });

        c->set_output(0, neighbors_shape);

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
        shape_inference::ShapeHandle matrix_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &matrix_shape));

        shape_inference::ShapeHandle vector_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &vector_shape));

        // 确定输出形状为 [batch_size, num_cols]
        shape_inference::DimensionHandle batch_size = c->Dim(matrix_shape, 0);
        // shape_inference::DimensionHandle num_cols   = c->Dim(matrix_shape,
        // 1);
        c->set_output(0, c->Matrix(batch_size, 1));

        return Status::OK();
    });

REGISTER_OP("IndirectSortAndTopkOp")
    .Input("k: T")
    .Input("v: U")
    .Output("sorted_k: T")
    .Output("sorted_v: U")
    .Attr("T: {int32, int64 } = DT_INT32")
    .Attr("U: {float, double} = DT_FLOAT")
    .Attr("topk: int >= 0 = 1000")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_shape));
        c->set_output(0, input_shape);
        c->set_output(1, input_shape);
        return Status::OK();
    });

REGISTER_OP("ResultConstructOp")
    .Input("k: int32")
    .Input("v: float")
    .Output("results_labels: int32")
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
