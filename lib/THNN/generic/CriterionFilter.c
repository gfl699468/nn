#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/CriterionFilter.c"
#else
//TODO: ANY better way to fill 0 in gradInput instead of nested for loop?

void THNN_(CriterionFilter_updateGradInput)(
          THNNState *state,
          THIndexTensor *target,
          THTensor *gradInput,
          THIndexTensor *ignored_label)
{
  int n_dims = THTensor_(nDimension)(target);

  gradInput = THTensor_(newContiguous)(gradInput);
  target = THIndexTensor_(newContiguous)(target);
  ignored_label = THIndexTensor_(newContiguous)(ignored_label);

  real *gradInput_data = THTensor_(data)(gradInput);
  THIndex_t *target_data = THIndexTensor_(data)(target);
  THIndex_t *ignored_label_data = THIndexTensor_(data)(ignored_label);

  int i;
  int ignored_label_num = ignored_label_data[0];
  if (n_dims == 1) {
    int batch_size = THTensor_(size)(target, 0);
    int n_classes = THTensor_(size)(gradInput, 1);
    for (i = 0; i < batch_size; i++) {
      int j;
      if (target[i] == ignored_label_num) {
        for (j = 0; j < n_classes; j++) gradInput[(i * nClass) + j] = 0;
      }
    }
  } else if (n_dims == 2) {
    int H = THTensor_(size)(target, 0);
    int W = THTensor_(size)(target, 1);
    int n_classes = THTensor_(size)(gradInput, 1);
    #pragma omp parallel for
    for (i = 0; i < H; i++) {
      int j;
      for (j = 0; j < W; j++) {
        if (target[(i * W) + j] == ignored_label_num) {
          int l;
          for (l = 0; l < n_classes; l++) {
            gradInput[(l * W * H) + (i * W) + j] = 0;
          }
        }
      }
    }
  } else if (n_dims == 3) {
    int batch_size = THTensor_(size)(target, 0);
    int H = THTensor_(size)(target, 1);
    int W = THTensor_(size)(target, 2);
    int n_classes = THTensor_(size)(target, 1);
    #pragma omp parallel for
    for (i = 0; i < batch_size; i++) {
      int j;
      for (j = 0; j < H; j++) {
        int l;
        for (l = 0; l < W; l++) {
          if (target[(i * H * W) + (j * W) + l] == ignored_label_num) {
            int o;
            for (o = 0; o < n_classes; o++) {
              gradInput[(i * n_classes * H * W) + ( o * H * W) + (j * W) + l] = 0
            }
          }
        }
      }
    }
  } else {THError("Target tensor should be 1D~3D tensor!");}
}
