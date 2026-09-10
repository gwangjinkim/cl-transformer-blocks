#define _GNU_SOURCE
/* Pointer-only ABI around MLX C: Lisp never guesses a by-value struct ABI. */
#include <mlx/c/mlx.h>
#include "benchmark_api.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdatomic.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/stdio.h>
#elif defined(__linux__)
#include <linux/fs.h>
#endif

struct tb_tensor { mlx_array value; };
struct tb_context { mlx_device device; mlx_stream stream; mlx_stream io_stream; };
typedef struct { mlx_map_string_to_array arrays; mlx_map_string_to_array_iterator it; } tb_weights;
static _Atomic size_t live_tensors;
size_t tb_live_tensors(void) { return atomic_load(&live_tensors); }
static _Thread_local char error_message[2048];
static void capture_error(const char *message, void *unused) {
  (void)unused; snprintf(error_message, sizeof(error_message), "%s", message);
}
const char *tb_last_error(void) { return error_message; }
static int filesystem_error(const char *operation, const char *path) {
  int code = errno;
  snprintf(error_message, sizeof(error_message), "%s %s: %s",
           operation, path, strerror(code));
  return code ? code : 1;
}
int tb_directory_create(const char *path) {
  error_message[0] = '\0';
  if (!mkdir(path, 0777)) return 0;
  if (errno == EEXIST) return 1;
  filesystem_error("Cannot create publication staging directory", path);
  return -1;
}
int tb_directory_publish(const char *staging, const char *destination) {
  struct stat staging_stat;
  struct stat destination_stat;
  error_message[0] = '\0';
  if (lstat(staging, &staging_stat))
    return filesystem_error("Cannot inspect staged directory", staging);
  if (!S_ISDIR(staging_stat.st_mode)) {
    errno = ENOTDIR;
    return filesystem_error("Publication source is not a directory", staging);
  }
  if (lstat(destination, &destination_stat)) {
    if (errno != ENOENT)
      return filesystem_error("Cannot inspect publication destination", destination);
    if (rename(staging, destination))
      return filesystem_error("Cannot publish staged directory", destination);
    return 0;
  }
  if (!S_ISDIR(destination_stat.st_mode)) {
    errno = ENOTDIR;
    return filesystem_error("Publication destination is not a directory", destination);
  }
#if defined(__APPLE__)
  if (renamex_np(staging, destination, RENAME_SWAP))
    return filesystem_error("Cannot exchange publication directories", destination);
#elif defined(__linux__)
  if (renameat2(AT_FDCWD, staging, AT_FDCWD, destination, RENAME_EXCHANGE))
    return filesystem_error("Cannot exchange publication directories", destination);
#else
  errno = ENOTSUP;
  return filesystem_error("Atomic directory exchange is unsupported for", destination);
#endif
  return 0;
}
static tb_tensor *new_tensor(void) {
  tb_tensor *out = calloc(1, sizeof(*out));
  if (out) { out->value = mlx_array_new(); atomic_fetch_add(&live_tensors,1); }
  else capture_error("Tensor handle allocation failed", NULL);
  return out;
}
void tb_tensor_free(tb_tensor *a) { if (a) { mlx_array_free(a->value); free(a); atomic_fetch_sub(&live_tensors,1); } }
static tb_tensor *checked(tb_tensor *out, int status) {
  if (status) { tb_tensor_free(out); return NULL; } return out;
}
tb_context *tb_context_new(int gpu) {
  mlx_set_error_handler(capture_error, NULL, NULL);
  tb_context *ctx = calloc(1, sizeof(*ctx));
  if (!ctx) return NULL;
  ctx->device = mlx_device_new_type(gpu ? MLX_GPU : MLX_CPU, 0);
  bool available = false;
  if (mlx_device_is_available(&available, ctx->device) || !available) {
    capture_error("Requested MLX device is unavailable", NULL);
    mlx_device_free(ctx->device); free(ctx); return NULL;
  }
  ctx->stream = mlx_stream_new_device(ctx->device);
  mlx_device cpu=mlx_device_new_type(MLX_CPU,0);
  ctx->io_stream=mlx_stream_new_device(cpu);
  mlx_device_free(cpu);
  return ctx;
}
void tb_context_free(tb_context *ctx) {
  if (ctx) { mlx_synchronize(ctx->stream); mlx_stream_free(ctx->stream);
             mlx_synchronize(ctx->io_stream); mlx_stream_free(ctx->io_stream);
             mlx_device_free(ctx->device); free(ctx); }
}
int tb_synchronize(tb_context *ctx) { return mlx_synchronize(ctx->stream); }
tb_tensor *tb_tensor_float(const float *data, const int *shape, int rank) {
  tb_tensor *out = new_tensor(); if (!out) return NULL;
  mlx_array_free(out->value); out->value = mlx_array_new_data(data, shape, rank, MLX_FLOAT32);
  return checked(out, out->value.ctx ? 0 : 1);
}
tb_tensor *tb_tensor_int(const int *data, const int *shape, int rank) {
  tb_tensor *out = new_tensor(); if (!out) return NULL;
  mlx_array_free(out->value); out->value = mlx_array_new_data(data, shape, rank, MLX_INT32);
  return checked(out, out->value.ctx ? 0 : 1);
}
tb_tensor *tb_tensor_retain(tb_tensor *a) {
  tb_tensor *out = new_tensor(); if (!out) return NULL;
  return checked(out, mlx_array_set(&out->value, a->value));
}
int tb_tensor_rank(tb_tensor *a) { return (int)mlx_array_ndim(a->value); }
int tb_tensor_dim(tb_tensor *a, int axis) { return mlx_array_dim(a->value, axis); }
int tb_tensor_dtype(tb_tensor *a) { return (int)mlx_array_dtype(a->value); }
int tb_tensor_eval(tb_tensor *a) { return mlx_array_eval(a->value); }
int tb_tensor_copy_float(tb_context *c, tb_tensor *a, float *data, size_t size) {
  if (size != mlx_array_size(a->value) || mlx_array_dtype(a->value) != MLX_FLOAT32) {
    capture_error("Float copy requires matching size and float32 dtype", NULL); return 1;
  }
  mlx_array contiguous=mlx_array_new();
  int status=mlx_contiguous(&contiguous,a->value,false,c->stream);
  if(!status) status=mlx_array_eval(contiguous);
  if(!status) {
    const float *source=mlx_array_data_float32(contiguous);
    if(source) memcpy(data,source,size*sizeof(float)); else status=1;
  }
  mlx_array_free(contiguous); return status;
}
int tb_tensor_copy_int(tb_context *c, tb_tensor *a, int32_t *data, size_t size) {
  if (size != mlx_array_size(a->value) || mlx_array_dtype(a->value) != MLX_INT32) {
    capture_error("Int copy requires matching size and int32 dtype", NULL); return 1;
  }
  mlx_array contiguous=mlx_array_new();
  int status=mlx_contiguous(&contiguous,a->value,false,c->stream);
  if(!status) status=mlx_array_eval(contiguous);
  if(!status) {
    const int32_t *source=mlx_array_data_int32(contiguous);
    if(source) memcpy(data,source,size*sizeof(int32_t)); else status=1;
  }
  mlx_array_free(contiguous); return status;
}
#define BINARY(NAME, OP) \
 tb_tensor *NAME(tb_context *c, tb_tensor *a, tb_tensor *b) { \
 tb_tensor *r = new_tensor(); if (!r) return NULL; \
 return checked(r, OP(&r->value, a->value, b->value, c->stream)); }
BINARY(tb_matmul, mlx_matmul)
BINARY(tb_add, mlx_add)
BINARY(tb_multiply, mlx_multiply)
BINARY(tb_subtract, mlx_subtract)
BINARY(tb_divide, mlx_divide)
#define UNARY(NAME, OP) \
 tb_tensor *NAME(tb_context *c, tb_tensor *a) { \
 tb_tensor *r = new_tensor(); if (!r) return NULL; \
 return checked(r, OP(&r->value, a->value, c->stream)); }
UNARY(tb_sigmoid, mlx_sigmoid)
UNARY(tb_sqrt, mlx_sqrt)
UNARY(tb_tanh, mlx_tanh)
UNARY(tb_erf, mlx_erf)

tb_tensor *tb_cast_float(tb_context *c, tb_tensor *a) {
 tb_tensor *r = new_tensor(); if (!r) return NULL;
 return checked(r, mlx_astype(&r->value, a->value, MLX_FLOAT32, c->stream));
}
tb_tensor *tb_cast_float16(tb_context *c, tb_tensor *a) {
 tb_tensor *r = new_tensor(); if (!r) return NULL;
 return checked(r, mlx_astype(&r->value, a->value, MLX_FLOAT16, c->stream));
}
tb_tensor *tb_reshape(tb_context *c, tb_tensor *a, const int *shape, int rank) {
 tb_tensor *r = new_tensor(); if (!r) return NULL;
 return checked(r, mlx_reshape(&r->value, a->value, shape, rank, c->stream));
}
tb_tensor *tb_permute(tb_context *c, tb_tensor *a, const int *axes, int rank) {
 tb_tensor *r = new_tensor(); if (!r) return NULL;
 return checked(r, mlx_transpose_axes(&r->value, a->value, axes, rank, c->stream));
}
tb_tensor *tb_take(tb_context *c, tb_tensor *a, tb_tensor *ids, int axis) {
 tb_tensor *r = new_tensor(); if (!r) return NULL;
 return checked(r, mlx_take_axis(&r->value, a->value, ids->value, axis, c->stream));
}
tb_tensor *tb_slice(tb_context *c, tb_tensor *a, const int *start, const int *stop, int rank) {
 int *strides = malloc(rank * sizeof(int));
 for (int i=0;i<rank;i++) strides[i]=1;
 tb_tensor *r = new_tensor(); if (!r) { free(strides); return NULL; }
 int status = mlx_slice(&r->value, a->value,start,rank,stop,rank,strides,rank,c->stream);
 free(strides); return checked(r,status);
}
tb_tensor *tb_zeros(tb_context *c, const int *shape, int rank) {
 tb_tensor *r = new_tensor(); if (!r) return NULL;
 return checked(r, mlx_zeros(&r->value, shape, rank, MLX_FLOAT32, c->stream));
}
tb_tensor *tb_slice_update(tb_context *c, tb_tensor *src, tb_tensor *update,
                           const int *start, const int *stop, int rank) {
 int *strides = malloc(rank * sizeof(int));
 if (!strides) { capture_error("Slice-update stride allocation failed", NULL); return NULL; }
 for (int i=0;i<rank;i++) strides[i]=1;
 tb_tensor *r = new_tensor(); if (!r) { free(strides); return NULL; }
 int status = mlx_slice_update(&r->value, src->value, update->value,
                               start, rank, stop, rank, strides, rank, c->stream);
 free(strides); return checked(r,status);
}
tb_tensor *tb_concat(tb_context *c, tb_tensor *a, tb_tensor *b, int axis) {
 mlx_array data[] = {a->value,b->value};
 mlx_vector_array vec=mlx_vector_array_new_data(data,2);
 tb_tensor *r=new_tensor(); if (!r) { mlx_vector_array_free(vec); return NULL; }
 int status=mlx_concatenate_axis(&r->value,vec,axis,c->stream);
 mlx_vector_array_free(vec); return checked(r,status);
}
tb_tensor *tb_rms_norm(tb_context *c, tb_tensor *a, tb_tensor *weight, float eps) {
 tb_tensor *r=new_tensor(); if (!r) return NULL;
 return checked(r,mlx_fast_rms_norm(&r->value,a->value,weight->value,eps,c->stream));
}
tb_tensor *tb_rope(tb_context *c, tb_tensor *a, int dims, float base, int offset) {
 tb_tensor *r=new_tensor(); if (!r) return NULL;
 mlx_optional_float theta={base,true};
 return checked(r,mlx_fast_rope(&r->value,a->value,dims,false,theta,1.0f,offset,
                              (mlx_array){NULL},c->stream));
}
tb_tensor *tb_attention(tb_context *c, tb_tensor *q, tb_tensor *k, tb_tensor *v,
                        float scale, tb_tensor *mask) {
 tb_tensor *r=new_tensor(); if (!r) return NULL;
 return checked(r,mlx_fast_scaled_dot_product_attention(&r->value,q->value,k->value,
   v->value,scale,mask ? "" : "causal",mask ? mask->value : (mlx_array){NULL},
   (mlx_array){NULL},false,c->stream));
}
tb_tensor *tb_cross_entropy(tb_context *c, tb_tensor *logits, tb_tensor *labels) {
 tb_tensor *r=new_tensor(); if (!r) return NULL;
 int status=mlx_fast_cross_entropy(&r->value,logits->value,labels->value,c->stream);
 if (!status) status=mlx_mean(&r->value,r->value,false,c->stream);
 return checked(r,status);
}
/* Inputs are selected [tokens,vocabulary] logits. Stable log probabilities,
   token-mean KL, and T^2 scaling; teacher never participates in differentiation. */
tb_tensor *tb_distillation_loss(tb_context *c, tb_tensor *student, tb_tensor *teacher, float temperature) {
 tb_tensor *r=new_tensor(); if (!r) return NULL;
 mlx_array s=mlx_array_new(), t=mlx_array_new(), z=mlx_array_new();
 mlx_array p=mlx_array_new(), term=mlx_array_new();
 mlx_array temp=mlx_array_new_float(temperature);
 mlx_array scale=mlx_array_new_float(temperature*temperature/mlx_array_dim(student->value,0));
 int status=mlx_divide(&s,student->value,temp,c->stream);
 if (!status) status=mlx_stop_gradient(&t,teacher->value,c->stream);
 if (!status) status=mlx_divide(&t,t,temp,c->stream);
 if (!status) status=mlx_logsumexp_axis(&z,s,-1,true,c->stream);
 if (!status) status=mlx_subtract(&s,s,z,c->stream);
 if (!status) status=mlx_logsumexp_axis(&z,t,-1,true,c->stream);
 if (!status) status=mlx_subtract(&t,t,z,c->stream);
 if (!status) status=mlx_exp(&p,t,c->stream);
 if (!status) status=mlx_subtract(&term,t,s,c->stream);
 if (!status) status=mlx_multiply(&term,p,term,c->stream);
 if (!status) status=mlx_sum(&r->value,term,false,c->stream);
 if (!status) status=mlx_multiply(&r->value,r->value,scale,c->stream);
 mlx_array_free(s); mlx_array_free(t); mlx_array_free(z); mlx_array_free(p);
 mlx_array_free(term); mlx_array_free(temp); mlx_array_free(scale);
 return checked(r,status);
}

/* Convert selected dense teacher logits into a fixed-temperature distribution over
   K explicit classes and one exact aggregate tail event. */
int tb_distillation_topk_target(tb_context *c, tb_tensor *teacher, int k, float temperature,
                                tb_tensor **top_log_probs_out, tb_tensor **indices_out,
                                tb_tensor **tail_log_prob_out) {
 *top_log_probs_out=NULL; *indices_out=NULL; *tail_log_prob_out=NULL;
 if(mlx_array_ndim(teacher->value)!=2 || mlx_array_dtype(teacher->value)!=MLX_FLOAT32 || k<=0 ||
    k>=mlx_array_dim(teacher->value,1) || !isfinite(temperature) || temperature<=0) {
   capture_error("Top-k target requires rank-two logits, 0 < k < vocabulary, and positive temperature",NULL);
   return 1;
 }
 tb_tensor *top=new_tensor(), *indices=new_tensor(), *tail=new_tensor();
 if(!top || !indices || !tail) {
   tb_tensor_free(top); tb_tensor_free(indices); tb_tensor_free(tail); return 1;
 }
 int rows=mlx_array_dim(teacher->value,0), vocabulary=mlx_array_dim(teacher->value,1);
 int start[2]={0,vocabulary-k}, stop[2]={rows,vocabulary}, strides[2]={1,1};
 int top_shape[2]={rows,k};
 mlx_array log_probs=mlx_array_new(), normalizer=mlx_array_new();
 mlx_array partition=mlx_array_new(), top_indices=mlx_array_new();
 mlx_array masked=mlx_array_new(), negative_values=mlx_array_new();
 mlx_array temp=mlx_array_new_float(temperature), negative=mlx_array_new_float(-INFINITY);
 int status=mlx_divide(&log_probs,teacher->value,temp,c->stream);
 if(!status) status=mlx_logsumexp_axis(&normalizer,log_probs,-1,true,c->stream);
 if(!status) status=mlx_subtract(&log_probs,log_probs,normalizer,c->stream);
 if(!status) status=mlx_argpartition_axis(&partition,teacher->value,vocabulary-k,-1,c->stream);
 if(!status) status=mlx_slice(&top_indices,partition,start,2,stop,2,strides,2,c->stream);
 if(!status) status=mlx_astype(&indices->value,top_indices,MLX_INT32,c->stream);
 if(!status) status=mlx_take_along_axis(&top->value,log_probs,indices->value,-1,c->stream);
 if(!status) status=mlx_full(&negative_values,top_shape,2,negative,MLX_FLOAT32,c->stream);
 if(!status) status=mlx_put_along_axis(&masked,log_probs,indices->value,negative_values,-1,c->stream);
 if(!status) status=mlx_logsumexp_axis(&tail->value,masked,-1,true,c->stream);
 mlx_array_free(log_probs); mlx_array_free(normalizer); mlx_array_free(partition);
 mlx_array_free(top_indices); mlx_array_free(masked); mlx_array_free(negative_values);
 mlx_array_free(temp); mlx_array_free(negative);
 if(status) { tb_tensor_free(top); tb_tensor_free(indices); tb_tensor_free(tail); return status; }
 *top_log_probs_out=top; *indices_out=indices; *tail_log_prob_out=tail;
 return 0;
}

/* KL between coarsened teacher/student distributions. Teacher values are normalized
   again after storage conversion; student tail probability is computed by masking
   the K explicit classes before a stable log-sum-exp. */
tb_tensor *tb_distillation_topk_loss(tb_context *c, tb_tensor *student,
                                     tb_tensor *top_log_probs, tb_tensor *indices,
                                     tb_tensor *tail_log_prob, float temperature) {
 if(mlx_array_ndim(student->value)!=2 || mlx_array_dtype(student->value)!=MLX_FLOAT32 ||
    mlx_array_ndim(top_log_probs->value)!=2 ||
    mlx_array_ndim(indices->value)!=2 || mlx_array_ndim(tail_log_prob->value)!=2 ||
    mlx_array_dtype(top_log_probs->value)!=MLX_FLOAT32 ||
    mlx_array_dtype(indices->value)!=MLX_INT32 ||
    mlx_array_dtype(tail_log_prob->value)!=MLX_FLOAT32 ||
    mlx_array_dim(student->value,0)!=mlx_array_dim(top_log_probs->value,0) ||
    mlx_array_dim(indices->value,0)!=mlx_array_dim(top_log_probs->value,0) ||
    mlx_array_dim(indices->value,1)!=mlx_array_dim(top_log_probs->value,1) ||
    mlx_array_dim(tail_log_prob->value,0)!=mlx_array_dim(top_log_probs->value,0) ||
    mlx_array_dim(tail_log_prob->value,1)!=1 ||
    mlx_array_dim(top_log_probs->value,1)<=0 ||
    mlx_array_dim(top_log_probs->value,1)>=mlx_array_dim(student->value,1) ||
    !isfinite(temperature) || temperature<=0) {
   capture_error("Top-k distillation tensors or temperature are incompatible",NULL); return NULL;
 }
 tb_tensor *r=new_tensor(); if(!r) return NULL;
 int rows=mlx_array_dim(top_log_probs->value,0), k=mlx_array_dim(top_log_probs->value,1);
 int top_shape[2]={rows,k};
 mlx_array s=mlx_array_new(), z=mlx_array_new(), student_top=mlx_array_new();
 mlx_array masked=mlx_array_new(), student_tail=mlx_array_new();
 mlx_array teacher_top=mlx_array_new(), teacher_tail=mlx_array_new();
 mlx_array teacher_all=mlx_array_new(), teacher_z=mlx_array_new();
 mlx_array q_top=mlx_array_new(), q_tail=mlx_array_new();
 mlx_array term=mlx_array_new(), tail_term=mlx_array_new();
 mlx_array top_sum=mlx_array_new(), tail_sum=mlx_array_new(), total=mlx_array_new();
 mlx_array negative_values=mlx_array_new();
 mlx_array temp=mlx_array_new_float(temperature);
 mlx_array negative=mlx_array_new_float(-INFINITY);
 mlx_array scale=mlx_array_new_float(temperature*temperature/rows);
 int status=mlx_stop_gradient(&teacher_top,top_log_probs->value,c->stream);
 if(!status) status=mlx_stop_gradient(&teacher_tail,tail_log_prob->value,c->stream);
 if(!status) {
   mlx_array target_data[]={teacher_top,teacher_tail};
   mlx_vector_array target=mlx_vector_array_new_data(target_data,2);
   status=mlx_concatenate_axis(&teacher_all,target,-1,c->stream);
   mlx_vector_array_free(target);
 }
 if(!status) status=mlx_logsumexp_axis(&teacher_z,teacher_all,-1,true,c->stream);
 if(!status) status=mlx_subtract(&teacher_top,teacher_top,teacher_z,c->stream);
 if(!status) status=mlx_subtract(&teacher_tail,teacher_tail,teacher_z,c->stream);
 if(!status) status=mlx_divide(&s,student->value,temp,c->stream);
 if(!status) status=mlx_logsumexp_axis(&z,s,-1,true,c->stream);
 if(!status) status=mlx_subtract(&s,s,z,c->stream);
 if(!status) status=mlx_take_along_axis(&student_top,s,indices->value,-1,c->stream);
 if(!status) status=mlx_full(&negative_values,top_shape,2,negative,MLX_FLOAT32,c->stream);
 if(!status) status=mlx_put_along_axis(&masked,s,indices->value,negative_values,-1,c->stream);
 if(!status) status=mlx_logsumexp_axis(&student_tail,masked,-1,true,c->stream);
 if(!status) status=mlx_exp(&q_top,teacher_top,c->stream);
 if(!status) status=mlx_exp(&q_tail,teacher_tail,c->stream);
 if(!status) status=mlx_subtract(&term,teacher_top,student_top,c->stream);
 if(!status) status=mlx_multiply(&term,q_top,term,c->stream);
 if(!status) status=mlx_subtract(&tail_term,teacher_tail,student_tail,c->stream);
 if(!status) status=mlx_multiply(&tail_term,q_tail,tail_term,c->stream);
 if(!status) status=mlx_sum(&top_sum,term,false,c->stream);
 if(!status) status=mlx_sum(&tail_sum,tail_term,false,c->stream);
 if(!status) status=mlx_add(&total,top_sum,tail_sum,c->stream);
 if(!status) status=mlx_multiply(&r->value,total,scale,c->stream);
 mlx_array_free(s); mlx_array_free(z); mlx_array_free(student_top); mlx_array_free(masked);
 mlx_array_free(student_tail); mlx_array_free(teacher_top); mlx_array_free(teacher_tail);
 mlx_array_free(teacher_all); mlx_array_free(teacher_z); mlx_array_free(q_top); mlx_array_free(q_tail);
 mlx_array_free(term); mlx_array_free(tail_term); mlx_array_free(top_sum); mlx_array_free(tail_sum);
 mlx_array_free(total); mlx_array_free(negative_values); mlx_array_free(temp);
 mlx_array_free(negative); mlx_array_free(scale);
 return checked(r,status);
}
tb_weights *tb_weights_load(tb_context *c, const char *path) {
 tb_weights *w=calloc(1,sizeof(*w)); if (!w) return NULL;
 w->arrays=mlx_map_string_to_array_new();
 mlx_map_string_to_string metadata=mlx_map_string_to_string_new();
 int status=mlx_load_safetensors(&w->arrays,&metadata,path,c->io_stream);
 mlx_map_string_to_string_free(metadata);
 if (status) { mlx_map_string_to_array_free(w->arrays); free(w); return NULL; }
 w->it=mlx_map_string_to_array_iterator_new(w->arrays); return w;
}
int tb_weights_next(tb_weights *w, const char **key, tb_tensor **value) {
 tb_tensor *r=new_tensor(); if (!r) return 1;
 int status=mlx_map_string_to_array_iterator_next(key,&r->value,w->it);
 if (status) tb_tensor_free(r); else *value=r;
 return status;
}
void tb_weights_free(tb_weights *w) {
 if (w) { mlx_map_string_to_array_iterator_free(w->it);
          mlx_map_string_to_array_free(w->arrays); free(w); }
}
int tb_weights_save(const char *path, const char **names, tb_tensor **values, int count) {
 mlx_map_string_to_array arrays=mlx_map_string_to_array_new();
 mlx_map_string_to_string metadata=mlx_map_string_to_string_new();
 int status=mlx_map_string_to_string_insert(metadata,"format","pt");
 for(int i=0;i<count && !status;i++)
   status=mlx_map_string_to_array_insert(arrays,names[i],values[i]->value);
 if(!status) status=mlx_save_safetensors(path,arrays,metadata);
 mlx_map_string_to_array_free(arrays); mlx_map_string_to_string_free(metadata);
 return status;
}

int tb_next_token(tb_context *c, tb_tensor *logits, int *token) {
 if (mlx_array_ndim(logits->value)!=3 || mlx_array_dim(logits->value,0)!=1) {
   capture_error("Greedy generation requires batch size one",NULL); return 1;
 }
 mlx_array indices=mlx_array_new();
 int status=mlx_argmax_axis(&indices,logits->value,-1,false,c->stream);
 if(!status) status=mlx_array_eval(indices);
 if(!status) {
   const uint32_t *data=mlx_array_data_uint32(indices);
   if(data) *token=(int)data[mlx_array_size(indices)-1]; else status=1;
 }
 mlx_array_free(indices); return status;
}

/* Differentiation traces the Lisp-defined model on this calling thread. No callback
   is retained after this function returns; no Lisp condition crosses the C stack. */
typedef int (*tb_loss_callback)(tb_tensor **out, tb_tensor **inputs, int count, void *payload);
typedef struct { tb_loss_callback callback; void *payload; } tb_loss_payload;
static int invoke_loss(mlx_vector_array *out, const mlx_vector_array inputs, void *payload) {
 tb_loss_payload *p=payload;
 int count=(int)mlx_vector_array_size(inputs), status=0;
 tb_tensor **handles=calloc(count,sizeof(*handles));
 if(!handles) return 1;
 for(int i=0;i<count;i++) {
   handles[i]=new_tensor();
   if(!handles[i] || mlx_vector_array_get(&handles[i]->value,inputs,i)) { status=1; break; }
 }
 tb_tensor *loss=NULL;
 if(!status) status=p->callback(&loss,handles,count,p->payload);
 if(!status && loss) status=mlx_vector_array_set_value(out,loss->value);
 else if(!loss) status=1;
 tb_tensor_free(loss);
 for(int i=0;i<count;i++) tb_tensor_free(handles[i]);
 free(handles); return status;
}
int tb_value_and_grad(tb_loss_callback callback, void *payload, tb_tensor **inputs,
                      int count, tb_tensor **loss_out, tb_tensor **grads_out) {
 if(count<=0) return 1;
 int status=0;
 *loss_out=NULL;
 for(int i=0;i<count;i++) grads_out[i]=NULL;
 tb_loss_payload p={callback,payload};
 mlx_vector_array x=mlx_vector_array_new(), loss=mlx_vector_array_new(), grads=mlx_vector_array_new();
 mlx_closure fun=mlx_closure_new_func_payload(invoke_loss,&p,NULL);
 mlx_closure_value_and_grad vg=mlx_closure_value_and_grad_new();
 int *argnums=malloc(count*sizeof(int));
 if(!argnums) { status=1; goto cleanup; }
 *loss_out=NULL;
 for(int i=0;i<count;i++) { grads_out[i]=NULL; argnums[i]=i;
   if(mlx_vector_array_append_value(x,inputs[i]->value)) { status=1; goto cleanup; }
 }
 status=mlx_value_and_grad(&vg,fun,argnums,count);
 if(!status) status=mlx_closure_value_and_grad_apply(&loss,&grads,vg,x);
 if(!status) status=mlx_eval(loss);
 if(!status) status=mlx_eval(grads);
 if(!status && ((int)mlx_vector_array_size(grads)!=count || mlx_vector_array_size(loss)!=1)) status=1;
 if(!status) {
   *loss_out=new_tensor();
   if(!*loss_out) status=1;
   else status=mlx_vector_array_get(&(*loss_out)->value,loss,0);
   for(int i=0;i<count && !status;i++) {
     grads_out[i]=new_tensor();
     if(!grads_out[i]) status=1;
     else status=mlx_vector_array_get(&grads_out[i]->value,grads,i);
   }
 }
cleanup:
 if(status) {
   tb_tensor_free(*loss_out); *loss_out=NULL;
   for(int i=0;i<count;i++) { tb_tensor_free(grads_out[i]); grads_out[i]=NULL; }
 }
 free(argnums); mlx_closure_value_and_grad_free(vg); mlx_closure_free(fun);
 mlx_vector_array_free(x); mlx_vector_array_free(loss); mlx_vector_array_free(grads);
 return status;
}


tb_tensor *tb_sum(tb_context *c, tb_tensor *a) {
 tb_tensor *r=new_tensor(); if(!r) return NULL;
 return checked(r,mlx_sum(&r->value,a->value,false,c->stream));
}
int tb_memory_stats(tb_context *c, size_t *active, size_t *cache, size_t *peak) {
 if(mlx_synchronize(c->stream) || mlx_synchronize(c->io_stream)) return 1;
 return mlx_get_active_memory(active) || mlx_get_cache_memory(cache) || mlx_get_peak_memory(peak);
}
int tb_reset_peak_memory(tb_context *c) {
 if(mlx_synchronize(c->stream) || mlx_synchronize(c->io_stream)) return 1;
 return mlx_reset_peak_memory();
}
int tb_tensor_finite(tb_context *c, tb_tensor *a, int *finite) {
 mlx_array result=mlx_array_new();
 int status=mlx_isfinite(&result,a->value,c->stream);
 if(!status) status=mlx_all(&result,result,false,c->stream);
 if(!status) status=mlx_array_eval(result);
 if(!status) { const bool *p=mlx_array_data_bool(result); if(p) *finite=*p; else status=1; }
 mlx_array_free(result); return status;
}


tb_tensor *tb_layer_norm(tb_context *c, tb_tensor *a, tb_tensor *weight, tb_tensor *bias, float eps) {
 tb_tensor *r=new_tensor(); if(!r) return NULL;
 return checked(r,mlx_fast_layer_norm(&r->value,a->value,weight->value,bias->value,eps,c->stream));
}
