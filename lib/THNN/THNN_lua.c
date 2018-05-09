#ifdef __cplusplus
#define THNN_EXTERNC extern "C"
#else
#define THNN_EXTERNC extern
#endif

#ifdef _WIN32
#ifdef THNN_EXPORTS
#define THNN_API THNN_EXTERNC __declspec(dllexport)
#else
#define THNN_API THNN_EXTERNC __declspec(dllimport)
#endif
#else
#define THNN_API THNN_EXTERNC
#endif

#ifdef __cplusplus
extern "C" {
#endif
#include <lauxlib.h>
#include <lua.h>
#include <lualib.h>

THNN_API int luaopen_THNN(lua_State *L);

#ifdef __cplusplus
}
#endif

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "TH.h"
#include "THMath.h"
#include "THNN.h"
#include "luaT.h"

#ifndef _CWRAP_STR_ARG_TYPES_4821726c1947cdf3eebacade98173939
#define _CWRAP_STR_ARG_TYPES_4821726c1947cdf3eebacade98173939
#include "string.h"
static void str_arg_types(lua_State *L, char *buf, int n) {
  int i;
  int nargs = lua_gettop(L);
  if (nargs == 0) {
    snprintf(buf, n, "no arguments provided");
    return;
  }
  for (i = 1; i <= nargs; i++) {
    int l;
    const char *torch_type = luaT_typename(L, i);
    if (torch_type && !strncmp(torch_type, "torch.", 6))
      torch_type += 6;
    if (torch_type)
      l = snprintf(buf, n, "%s ", torch_type);
    else if (lua_isnil(L, i))
      l = snprintf(buf, n, "%s ", "nil");
    else if (lua_isboolean(L, i))
      l = snprintf(buf, n, "%s ", "boolean");
    else if (lua_isnumber(L, i))
      l = snprintf(buf, n, "%s ", "number");
    else if (lua_isstring(L, i))
      l = snprintf(buf, n, "%s ", "string");
    else if (lua_istable(L, i))
      l = snprintf(buf, n, "%s ", "table");
    else if (lua_isuserdata(L, i))
      l = snprintf(buf, n, "%s ", "userdata");
    else
      l = snprintf(buf, n, "%s ", "???");
    if (l >= n)
      return;
    buf += l;
    n -= l;
  }
}
#endif

#define MAKE_FUNC_TT(name)                                                     \
  static int torch_Float_##name(lua_State *L) {                                \
    int narg = lua_gettop(L);                                                  \
    THNNState *arg1 = NULL;                                                    \
    THFloatTensor *arg2 = NULL;                                                \
    THFloatTensor *arg3 = NULL;                                                \
    if (narg == 2 && (arg2 = luaT_toudata(L, 1, "torch.FloatTensor")) &&       \
        (arg3 = luaT_toudata(L, 2, "torch.FloatTensor"))) {                    \
    } else {                                                                   \
      char type_buf[512];                                                      \
      str_arg_types(L, type_buf, 512);                                         \
      luaL_error(L,                                                            \
                 "invalid arguments: %s\nexpected arguments: FloatTensor "     \
                 "FloatTensor",                                                \
                 type_buf);                                                    \
    }                                                                          \
    THNN_Float##name(arg1, arg2, arg3);                                        \
    return 0;                                                                  \
  }                                                                            \
  static int torch_Double_##name(lua_State *L) {                               \
    int narg = lua_gettop(L);                                                  \
    THNNState *arg1 = NULL;                                                    \
    THDoubleTensor *arg2 = NULL;                                               \
    THDoubleTensor *arg3 = NULL;                                               \
    if (narg == 2 && (arg2 = luaT_toudata(L, 1, "torch.DoubleTensor")) &&      \
        (arg3 = luaT_toudata(L, 2, "torch.DoubleTensor"))) {                   \
    } else {                                                                   \
      char type_buf[512];                                                      \
      str_arg_types(L, type_buf, 512);                                         \
      luaL_error(L,                                                            \
                 "invalid arguments: %s\nexpected arguments: DoubleTensor "    \
                 "DoubleTensor",                                               \
                 type_buf);                                                    \
    }                                                                          \
    THNN_Double##name(arg1, arg2, arg3);                                       \
    return 0;                                                                  \
  }

#define MAKE_FUNC_TTT(name)                                                    \
  static int torch_Float_##name(lua_State *L) {                                \
    int narg = lua_gettop(L);                                                  \
    THNNState *arg1 = NULL;                                                    \
    THFloatTensor *arg2 = NULL;                                                \
    THFloatTensor *arg3 = NULL;                                                \
    THFloatTensor *arg4 = NULL;                                                \
    if (narg == 3 && (arg2 = luaT_toudata(L, 1, "torch.FloatTensor")) &&       \
        (arg3 = luaT_toudata(L, 2, "torch.FloatTensor")) &&                    \
        (arg4 = luaT_toudata(L, 3, "torch.FloatTensor"))) {                    \
    } else {                                                                   \
      char type_buf[512];                                                      \
      str_arg_types(L, type_buf, 512);                                         \
      luaL_error(L,                                                            \
                 "invalid arguments: %s\nexpected arguments: FloatTensor "     \
                 "FloatTensor FloatTensor",                                    \
                 type_buf);                                                    \
    }                                                                          \
    THNN_Float##name(arg1, arg2, arg3, arg4);                                  \
    return 0;                                                                  \
  }                                                                            \
  static int torch_Double_##name(lua_State *L) {                               \
    int narg = lua_gettop(L);                                                  \
    THNNState *arg1 = NULL;                                                    \
    THDoubleTensor *arg2 = NULL;                                               \
    THDoubleTensor *arg3 = NULL;                                               \
    THDoubleTensor *arg4 = NULL;                                               \
    if (narg == 3 && (arg2 = luaT_toudata(L, 1, "torch.DoubleTensor")) &&      \
        (arg3 = luaT_toudata(L, 2, "torch.DoubleTensor")) &&                   \
        (arg4 = luaT_toudata(L, 3, "torch.DoubleTensor"))) {                   \
    } else {                                                                   \
      char type_buf[512];                                                      \
      str_arg_types(L, type_buf, 512);                                         \
      luaL_error(L,                                                            \
                 "invalid arguments: %s\nexpected arguments: DoubleTensor "    \
                 "DoubleTensor DoubleTensor",                                  \
                 type_buf);                                                    \
    }                                                                          \
    THNN_Double##name(arg1, arg2, arg3, arg4);                                 \
    return 0;                                                                  \
  }

#define MAKE_FUNC_TTTB(name)                                                   \
  static int torch_Float_##name(lua_State *L) {                                \
    int narg = lua_gettop(L);                                                  \
    THNNState *arg1 = NULL;                                                    \
    THFloatTensor *arg2 = NULL;                                                \
    THFloatTensor *arg3 = NULL;                                                \
    THFloatTensor *arg4 = NULL;                                                \
    int arg5 = 0;                                                              \
    if (narg == 4 && (arg2 = luaT_toudata(L, 1, "torch.FloatTensor")) &&       \
        (arg3 = luaT_toudata(L, 2, "torch.FloatTensor")) &&                    \
        (arg4 = luaT_toudata(L, 3, "torch.FloatTensor")) &&                    \
        lua_isboolean(L, 4)) {                                                 \
      arg5 = lua_toboolean(L, 4);                                              \
    } else {                                                                   \
      char type_buf[512];                                                      \
      str_arg_types(L, type_buf, 512);                                         \
      luaL_error(L,                                                            \
                 "invalid arguments: %s\nexpected arguments: FloatTensor "     \
                 "FloatTensor FloatTensor boolean",                            \
                 type_buf);                                                    \
    }                                                                          \
    THNN_Float##name(arg1, arg2, arg3, arg4, arg5);                            \
    return 0;                                                                  \
  }                                                                            \
  static int torch_Double_##name(lua_State *L) {                               \
    int narg = lua_gettop(L);                                                  \
    THNNState *arg1 = NULL;                                                    \
    THDoubleTensor *arg2 = NULL;                                               \
    THDoubleTensor *arg3 = NULL;                                               \
    THDoubleTensor *arg4 = NULL;                                               \
    int arg5 = 0;                                                              \
    if (narg == 4 && (arg2 = luaT_toudata(L, 1, "torch.DoubleTensor")) &&      \
        (arg3 = luaT_toudata(L, 2, "torch.DoubleTensor")) &&                   \
        (arg4 = luaT_toudata(L, 3, "torch.DoubleTensor")) &&                   \
        lua_isboolean(L, 4)) {                                                 \
      arg5 = lua_toboolean(L, 4);                                              \
    } else {                                                                   \
      char type_buf[512];                                                      \
      str_arg_types(L, type_buf, 512);                                         \
      luaL_error(L,                                                            \
                 "invalid arguments: %s\nexpected arguments: DoubleTensor "    \
                 "DoubleTensor DoubleTensor boolean",                          \
                 type_buf);                                                    \
    }                                                                          \
    THNN_Double##name(arg1, arg2, arg3, arg4, arg5);                           \
    return 0;                                                                  \
  }

MAKE_FUNC_TT(Abs_updateOutput)
MAKE_FUNC_TTT(Abs_updateGradInput)
MAKE_FUNC_TTTB(AbsCriterion_updateOutput)
MAKE_FUNC_TTTB(AbsCriterion_updateGradInput)

#undef MAKE_FUNC_TT
#define MAKE_FUNC_TT(name)                                                     \
  { #name, torch_Float_##name }
#undef MAKE_FUNC_TTT
#define MAKE_FUNC_TTT(name)                                                    \
  { #name, torch_Float_##name }
#undef MAKE_FUNC_TTTB
#define MAKE_FUNC_TTTB(name)                                                   \
  { #name, torch_Float_##name }

static const struct luaL_Reg torch_float_lib[] = {
    MAKE_FUNC_TT(Abs_updateOutput),
    MAKE_FUNC_TTT(Abs_updateGradInput),
    MAKE_FUNC_TTTB(AbsCriterion_updateOutput),
    MAKE_FUNC_TTTB(AbsCriterion_updateGradInput),
};

#undef MAKE_FUNC_TT
#define MAKE_FUNC_TT(name)                                                     \
  { #name, torch_Double_##name }
#undef MAKE_FUNC_TTT
#define MAKE_FUNC_TTT(name)                                                    \
  { #name, torch_Double_##name }
#undef MAKE_FUNC_TTTB
#define MAKE_FUNC_TTTB(name)                                                   \
  { #name, torch_Double_##name }

static const struct luaL_Reg torch_double_lib[] = {
    MAKE_FUNC_TT(Abs_updateOutput),
    MAKE_FUNC_TTT(Abs_updateGradInput),
    MAKE_FUNC_TTTB(AbsCriterion_updateOutput),
    MAKE_FUNC_TTTB(AbsCriterion_updateGradInput),
};

/* Adds the DoubleTensor functions to the THNN field
   in getmetatable(torch.DoubleTensor) */
static void torch_DoubleTensor_init(lua_State *L) {
  if (!luaT_pushmetatable(L, "torch.DoubleTensor"))
    return;

  /* register functions into the "torch" field of the tensor metaclass */
  lua_pushstring(L, "THNN");
  lua_newtable(L);
  luaT_setfuncs(L, torch_double_lib, 0);
  lua_rawset(L, -3);
  lua_pop(L, 1);
}

/* Adds the FloatTensor functions to the THNN field
in getmetatable(torch.FloatTensor) */
static void torch_FloatTensor_init(lua_State *L) {
  if (!luaT_pushmetatable(L, "torch.FloatTensor"))
    return;

  /* register functions into the "torch" field of the tensor metaclass */
  lua_pushstring(L, "THNN");
  lua_newtable(L);
  luaT_setfuncs(L, torch_float_lib, 0);
  lua_rawset(L, -3);
  lua_pop(L, 1);
}

int luaopen_THNN(lua_State *L) {
  fprintf(stderr, "Initializing THNN\n");
  torch_FloatTensor_init(L);
  torch_DoubleTensor_init(L);
  lua_createtable(L, 0, 0);
  fprintf(stdout, "THNN initialized successfully\n");
  return 1;
}
