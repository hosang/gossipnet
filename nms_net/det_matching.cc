#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using std::vector;

using namespace tensorflow;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::ShapeHandle;


REGISTER_OP("DetectionMatching")
    .Attr("T: {float}")
    .Input("iou: T")
    .Input("score: T")
    .Input("ignore: bool")
    .Output("labels: T")
    .Output("weights: T")
    .Output("assignment: int32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused, score_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &score_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      c->set_output(0, score_shape);
      c->set_output(1, score_shape);
      c->set_output(2, score_shape);
      return Status::OK();
    });


template <typename T>
class DetectionMatchingOp : public OpKernel {
 public:
  explicit DetectionMatchingOp(OpKernelConstruction* context) : OpKernel(context) {}

  template <typename T2>
  vector<size_t> argsort(const typename TTypes<T2>::ConstFlat &v) {
      vector<size_t> idx(v.dimension(0));
      iota(idx.begin(), idx.end(), 0);

      sort(idx.begin(), idx.end(),
              [&v](size_t i1, size_t i2) {return v(i1) < v(i2);});
      return idx;
  }

  void Compute(OpKernelContext* context) override {
    const T iou_thresh = 0.5;

    const Tensor& iou_tensor = context->input(0);
    auto ious = iou_tensor.tensor<T,2>();
    const Tensor& score_tensor = context->input(1);
    auto score = score_tensor.flat<T>();
    const Tensor& ignore_tensor = context->input(2);
    auto ignore = ignore_tensor.flat<bool>();


    auto det_order = argsort<T>(score);
    reverse(det_order.begin(), det_order.end());

    auto gt_order = argsort<bool>(ignore);

    // Create output tensors
    Tensor* label_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, score_tensor.shape(),
                                                     &label_tensor));
    auto labels = label_tensor->flat<T>();
    labels.setZero();

    Tensor* weight_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, score_tensor.shape(),
                                                     &weight_tensor));
    auto weights = weight_tensor->flat<T>();
    weights.setConstant(1);

    Tensor* assignment_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, score_tensor.shape(),
                                                     &assignment_tensor));
    auto assignments = assignment_tensor->flat<int32>();
    assignments.setConstant(-1);

    const int n_dets = ious.dimension(0);
    const int n_gt = ious.dimension(1);
    vector<uint8> is_matched(n_dets, 0);

    for (int _det_i = 0; _det_i < n_dets; ++_det_i) {
        const int det = det_order[_det_i];

        T iou = iou_thresh;
        int match = -1;
        for (int _gt_i = 0; _gt_i < n_gt; ++_gt_i) {
            const int gt = gt_order[_gt_i];

            // if this gt already matched, and not a crowd, continue
            if (is_matched[gt] && !ignore(gt)) continue;
            // if dt matched to reg gt, and on ignore gt, stop
            if (match > -1 && ignore(gt)) break;
            // continue to next gt unless better match made
            if (ious(det, gt) < iou) continue;

            // match successful and best so far, store appropriately
            iou = ious(det, gt);
            match = gt;
            is_matched[gt] = 1;
        }

        if (match > -1) {
            labels(det) = 1;
            assignments(det) = match;
            if (ignore(match)) {
                weights(det) = 0;
            }
        }
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("DetectionMatching")
                             .Device(DEVICE_CPU)
                             .TypeConstraint<float>("T"),
                        DetectionMatchingOp<float>);
