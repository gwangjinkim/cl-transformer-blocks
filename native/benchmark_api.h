/* Internal pointer-only ABI subset shared by the C baseline and the MLX bridge. */
#ifndef TB_BENCHMARK_API_H
#define TB_BENCHMARK_API_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct tb_tensor tb_tensor;
typedef struct tb_context tb_context;
const char *tb_last_error(void);
tb_context *tb_context_new(int gpu);
void tb_context_free(tb_context *context);
tb_tensor *tb_tensor_float(const float *data, const int *shape, int rank);
void tb_tensor_free(tb_tensor *tensor);
int tb_tensor_eval(tb_tensor *tensor);
int tb_tensor_copy_float(tb_context *context, tb_tensor *tensor, float *data, size_t size);
tb_tensor *tb_matmul(tb_context *context, tb_tensor *a, tb_tensor *b);
#ifdef __cplusplus
}
#endif
#endif
