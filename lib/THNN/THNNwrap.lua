-- Utility to generate Lua C api based
-- interface for torch.nn
-- Note that the wrap() calls were generated
-- using the tools/extract.lua utility
-- Some aspects of the interface needed changes
-- a) optional args handling - we use nil as optional indicator
-- b) IndexTensor args need noreadadd set to true to aoid adding -1
-- c) Index_t is mapped to int64_t rather than index as latter does adjustment by 1 that
--    breaks PReLU - need to check why Torch does this adjustment elsewhere but
--    here it causes problems

local wrap = require 'cwrap'

local interface = wrap.CInterface.new()
local argtypes = wrap.CInterface.argtypes

argtypes['ptrdiff_t'] = wrap.types.ptrdiff_t

for _,typename in ipairs({"FloatTensor", "DoubleTensor", "IndexTensor"}) do

   local torchname = typename == 'IndexTensor' and 'LongTensor' or typename
   argtypes[typename].check = function(arg, idx)
                 if arg.dim then
                    return string.format('(arg%d = luaT_toudata(L, %d, "torch.%s")) && (arg%d->nDimension == %d)', arg.i, idx, torchname, arg.i, arg.dim)
                 elseif arg.optional then
                    return string.format('((arg%d = luaT_toudata(L, %d, "torch.%s")), (lua_isnil(L, %d) ? 1 : arg%d != NULL))', arg.i, idx, torchname, idx, arg.i)
                 else
                    return string.format('(arg%d = luaT_toudata(L, %d, "torch.%s"))', arg.i, idx, torchname)                    
                 end
              end
end

local function interpretdefaultvalue(arg)
   local default = arg.default
   if type(default) == 'boolean' then
      if default then
         return '1'
      else
         return '0'
      end
   elseif type(default) == 'number' then
      return tostring(default)
   elseif type(default) == 'string' then
      return default
   elseif type(default) == 'function' then
      default = default(arg)
      assert(type(default) == 'string', 'a default function must return a string')
      return default
   elseif type(default) == 'nil' then
      return nil
   else
      error('unknown default type value')
   end   
end
argtypes.index = argtypes['int64_t']
argtypes.THNNState = {

   helpname = function(arg)
                 return "THNNState *"
              end,

   declare = function(arg)
                -- if it is a number we initialize here
                return string.format("THNNState *arg%d = NULL;", arg.i)
             end,

   check = function(arg, idx)
              return string.format("1", idx)
           end,

   read = function(arg, idx)
             return string.format("arg%d = NULL;", arg.i)
          end,

   init = function(arg)
             -- otherwise do it here
             return string.format("arg%d = NULL;", arg.i)
          end,

   carg = function(arg)
             return string.format('arg%d', arg.i)
          end,

   creturn = function(arg)
                return string.format('arg%d', arg.i)
             end,

   precall = function(arg)
                if arg.returned then
                   return string.format('lua_pushnil(L);')
                end
             end,

   postcall = function(arg)
                 if arg.creturned then
                    return string.format('lua_pushnil(L);')
                 end
              end
}

interface:print([[
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

THNN_API int luaopen_THNNx(lua_State *L);

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
]])

-- specific to torch: we generate a 'dispatch' function
-- first we create a helper function
-- note that it let the "torch" table on the stack
interface:print([[
static const void* torch_istensortype(lua_State *L, const char *tname)
{
  if(!tname)
    return NULL;

  if(!luaT_pushmetatable(L, tname))
    return NULL;

  lua_pushstring(L, "torch");
  lua_rawget(L, -2);
  if(lua_istable(L, -1))
    return tname;
  else
  {
    lua_pop(L, 2);
    return NULL;
  }

  return NULL;
}
]])

interface:print([[
static int torch_isnonemptytable(lua_State *L, int idx)
{
  int empty;
  if (!lua_istable(L, idx)) return 0;

  lua_rawgeti(L, idx, 1);
  empty = lua_isnil(L, -1);
  lua_pop(L, 1);
  return !empty;
}
]])


interface:print([[
static const void* torch_istensorarray(lua_State *L, int idx)
{
  const char* tname;
  int tensor_idx;
  if (!torch_isnonemptytable(L, idx)) return 0;

  lua_checkstack(L, 3);
  lua_rawgeti(L, idx, 1);
  tensor_idx = lua_gettop(L);
  tname = (torch_istensortype(L, luaT_typename(L, -1)));
  lua_remove(L, tensor_idx);
  return tname;
}
]])

interface:print('/* WARNING: autogenerated file */')
interface:print('')

local function wrap(...)
   local args = {...}
   -- interface
   interface:wrap(...)
end

local reals = {ByteTensor='uint8_t',
               CharTensor='int8_t',
               ShortTensor='int16_t',
               IntTensor='int32_t',
               LongTensor='int64_t',
               FloatTensor='float',
               HalfTensor='half',
               DoubleTensor='double'}

local accreals = {ByteTensor='int64_t',
               CharTensor='int64_t',
               ShortTensor='int64_t',
               IntTensor='int64_t',
               LongTensor='int64_t',
               FloatTensor='double',
               HalfTensor='float',
               DoubleTensor='double'}

for _,Tensor in ipairs({"FloatTensor", "DoubleTensor"}) do

   local real = reals[Tensor]
   local accreal = accreals[Tensor]
   local prefix = Tensor == "FloatTensor" and "Float" or "Double"

   function interface.luaname2wrapname(self, name)
      return string.format('torch_%s_%s', prefix, name)
   end

   local function cname(name)      
      return string.format('THNN_%s%s', prefix, name)
   end

   local function lastdim(argn)
      return function(arg)
                return string.format("TH%s_nDimension(%s)", Tensor, arg.args[argn]:carg())
             end
   end

   local function lastdimarray(argn)
      return function(arg)
                return string.format("TH%s_nDimension(arg%d_data[0])", Tensor, arg.args[argn].i)
             end
   end

   local THNNState = "THNNState"

   if Tensor == 'FloatTensor' or Tensor == 'DoubleTensor' then

     wrap("Abs_updateOutput",
        cname("Abs_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},})
     wrap("Abs_updateGradInput",
        cname("Abs_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},})
     wrap("AbsCriterion_updateOutput",
        cname("AbsCriterion_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="boolean"},})
     wrap("AbsCriterion_updateGradInput",
        cname("AbsCriterion_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="boolean"},})
     wrap("BCECriterion_updateOutput",
        cname("BCECriterion_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="boolean"},
        {name=Tensor, optional=true},})
     wrap("BCECriterion_updateGradInput",
        cname("BCECriterion_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="boolean"},
        {name=Tensor, optional=true},})
     wrap("ClassNLLCriterion_updateOutput",
        cname("ClassNLLCriterion_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name=Tensor},
        {name="boolean"},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name="int64_t"},})
     wrap("ClassNLLCriterion_updateGradInput",
        cname("ClassNLLCriterion_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name=Tensor},
        {name="boolean"},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name="int64_t"},})
     wrap("SpatialClassNLLCriterion_updateOutput",
        cname("SpatialClassNLLCriterion_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name=Tensor},
        {name="boolean"},
        {name=Tensor, optional=true},
        {name=Tensor},})
     wrap("SpatialClassNLLCriterion_updateGradInput",
        cname("SpatialClassNLLCriterion_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name=Tensor},
        {name="boolean"},
        {name=Tensor, optional=true},
        {name=Tensor},})
     wrap("ELU_updateOutput",
        cname("ELU_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},
        {name="boolean"},})
     wrap("ELU_updateGradInput",
        cname("ELU_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},
        {name="boolean"},})
     wrap("DistKLDivCriterion_updateOutput",
        cname("DistKLDivCriterion_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="boolean"},})
     wrap("DistKLDivCriterion_updateGradInput",
        cname("DistKLDivCriterion_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="boolean"},})
     wrap("GatedLinear_updateOutput",
        cname("GatedLinear_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name="int"},})
     wrap("GatedLinear_updateGradInput",
        cname("GatedLinear_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},})
     wrap("HardShrink_updateOutput",
        cname("HardShrink_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},})
     wrap("HardShrink_updateGradInput",
        cname("HardShrink_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},})
     wrap("HardTanh_updateOutput",
        cname("HardTanh_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},
        {name=accreal},
        {name="boolean"},})
     wrap("HardTanh_updateGradInput",
        cname("HardTanh_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},
        {name=accreal},
        {name="boolean"},})
     wrap("L1Cost_updateOutput",
        cname("L1Cost_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},})
     wrap("L1Cost_updateGradInput",
        cname("L1Cost_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor, optional=true},
        {name=Tensor},})
     wrap("LeakyReLU_updateOutput",
        cname("LeakyReLU_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},
        {name="boolean"},})
     wrap("LeakyReLU_updateGradInput",
        cname("LeakyReLU_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},
        {name="boolean"},})
     wrap("GRUFused_updateOutput",
        cname("GRUFused_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor, optional=true},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},})
     wrap("GRUFused_updateGradInput",
        cname("GRUFused_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},})
     wrap("LSTMFused_updateOutput",
        cname("LSTMFused_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor, optional=true},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},})
     wrap("LSTMFused_updateGradInput",
        cname("LSTMFused_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},})
     wrap("LogSigmoid_updateOutput",
        cname("LogSigmoid_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},})
     wrap("LogSigmoid_updateGradInput",
        cname("LogSigmoid_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},})
     wrap("LogSoftMax_updateOutput",
        cname("LogSoftMax_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},})
     wrap("LogSoftMax_updateGradInput",
        cname("LogSoftMax_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},})
     wrap("LookupTable_accGradParameters",
        cname("LookupTable_accGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name="IndexTensor", noreadadd=true},
        {name=Tensor},
        {name=Tensor},
        {name="IntTensor"},
        {name=Tensor, optional=true},
        {name="IndexTensor", noreadadd=true, optional=true},
        {name="boolean"},
        {name="int"},
        {name=accreal},})
     wrap("LookupTable_renorm",
        cname("LookupTable_renorm"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name="IndexTensor", noreadadd=true},
        {name=Tensor},
        {name=accreal},
        {name=accreal},})
     wrap("MarginCriterion_updateOutput",
        cname("MarginCriterion_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="boolean"},
        {name=accreal},})
     wrap("MarginCriterion_updateGradInput",
        cname("MarginCriterion_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="boolean"},
        {name=accreal},})
     wrap("SoftMarginCriterion_updateOutput",
        cname("SoftMarginCriterion_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="boolean"},})
     wrap("SoftMarginCriterion_updateGradInput",
        cname("SoftMarginCriterion_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="boolean"},})
     wrap("MSECriterion_updateOutput",
        cname("MSECriterion_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="boolean"},})
     wrap("MSECriterion_updateGradInput",
        cname("MSECriterion_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="boolean"},})
     wrap("MultiLabelMarginCriterion_updateOutput",
        cname("MultiLabelMarginCriterion_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name=Tensor},
        {name=Tensor},
        {name="boolean"},})
     wrap("MultiLabelMarginCriterion_updateGradInput",
        cname("MultiLabelMarginCriterion_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name=Tensor},
        {name=Tensor},
        {name="boolean"},})
     wrap("MultiMarginCriterion_updateOutput",
        cname("MultiMarginCriterion_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name=Tensor},
        {name="boolean"},
        {name="int"},
        {name=Tensor, optional=true},
        {name=accreal},})
     wrap("MultiMarginCriterion_updateGradInput",
        cname("MultiMarginCriterion_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name=Tensor},
        {name="boolean"},
        {name="int"},
        {name=Tensor, optional=true},
        {name=accreal},})
     wrap("PReLU_updateOutput",
        cname("PReLU_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="index"},})
     wrap("PReLU_updateGradInput",
        cname("PReLU_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="index"},})
     wrap("PReLU_accGradParameters",
        cname("PReLU_accGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="index"},
        {name=accreal},})
     wrap("Linear_updateOutput",
        cname("Linear_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},})
     wrap("Linear_updateGradInput",
        cname("Linear_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},})
     wrap("Linear_accGradParameters",
        cname("Linear_accGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},})
     wrap("RReLU_updateOutput",
        cname("RReLU_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},
        {name=accreal},
        {name="boolean"},
        {name="boolean"},
        {name="Generator"},})
     wrap("RReLU_updateGradInput",
        cname("RReLU_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},
        {name=accreal},
        {name="boolean"},
        {name="boolean"},})
     wrap("Sigmoid_updateOutput",
        cname("Sigmoid_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},})
     wrap("Sigmoid_updateGradInput",
        cname("Sigmoid_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},})
     wrap("SmoothL1Criterion_updateOutput",
        cname("SmoothL1Criterion_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="boolean"},})
     wrap("SmoothL1Criterion_updateGradInput",
        cname("SmoothL1Criterion_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="boolean"},})
     wrap("SoftMax_updateOutput",
        cname("SoftMax_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},})
     wrap("SoftMax_updateGradInput",
        cname("SoftMax_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},})
     wrap("SoftPlus_updateOutput",
        cname("SoftPlus_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},
        {name=accreal},})
     wrap("SoftPlus_updateGradInput",
        cname("SoftPlus_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},
        {name=accreal},})
     wrap("SoftShrink_updateOutput",
        cname("SoftShrink_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},})
     wrap("SoftShrink_updateGradInput",
        cname("SoftShrink_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},})
     wrap("IndexLinear_updateOutput",
        cname("IndexLinear_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name="IndexTensor", noreadadd=true},
        {name="int64_t"},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name="IndexTensor", noreadadd=true},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},})
     wrap("IndexLinear_accGradParameters",
        cname("IndexLinear_accGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name="IndexTensor", noreadadd=true},
        {name="int64_t"},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name="IndexTensor", noreadadd=true},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},
        {name=accreal},})
     wrap("IndexLinear_accUpdateGradParameters",
        cname("IndexLinear_accUpdateGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name="IndexTensor", noreadadd=true},
        {name="int64_t"},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name="IndexTensor", noreadadd=true},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},
        {name=accreal},})
     wrap("IndexLinear_updateParameters",
        cname("IndexLinear_updateParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name="IndexTensor", noreadadd=true},
        {name="int64_t"},
        {name=accreal},
        {name=accreal},})
     wrap("SparseLinear_updateOutput",
        cname("SparseLinear_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},})
     wrap("SparseLinear_accGradParameters",
        cname("SparseLinear_accGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},
        {name=accreal},})
     wrap("SparseLinear_zeroGradParameters",
        cname("SparseLinear_zeroGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},})
     wrap("SparseLinear_updateParameters",
        cname("SparseLinear_updateParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},})
     wrap("SparseLinear_legacyUpdateOutput",
        cname("SparseLinear_legacyUpdateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},})
     wrap("SparseLinear_legacyAccGradParameters",
        cname("SparseLinear_legacyAccGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},
        {name=accreal},})
     wrap("SparseLinear_legacyZeroGradParameters",
        cname("SparseLinear_legacyZeroGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},})
     wrap("SparseLinear_legacyUpdateParameters",
        cname("SparseLinear_legacyUpdateParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},})
     wrap("Sqrt_updateOutput",
        cname("Sqrt_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},})
     wrap("Sqrt_updateGradInput",
        cname("Sqrt_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},})
     wrap("Square_updateOutput",
        cname("Square_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},})
     wrap("Square_updateGradInput",
        cname("Square_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},})
     wrap("Tanh_updateOutput",
        cname("Tanh_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},})
     wrap("Tanh_updateGradInput",
        cname("Tanh_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},})
     wrap("Threshold_updateOutput",
        cname("Threshold_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},
        {name=accreal},
        {name="boolean"},})
     wrap("Threshold_updateGradInput",
        cname("Threshold_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},
        {name=accreal},
        {name="boolean"},})
     wrap("TemporalConvolution_updateOutput",
        cname("TemporalConvolution_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},
        {name="int"},})
     wrap("TemporalConvolution_updateGradInput",
        cname("TemporalConvolution_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},})
     wrap("TemporalConvolution_accGradParameters",
        cname("TemporalConvolution_accGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name=accreal},})
     wrap("TemporalMaxPooling_updateOutput",
        cname("TemporalMaxPooling_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name="int"},        {name="int"},})
     wrap("TemporalMaxPooling_updateGradInput",
        cname("TemporalMaxPooling_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name="int"},        {name="int"},})
     wrap("TemporalSubSampling_updateOutput",
        cname("TemporalSubSampling_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},})
     wrap("TemporalSubSampling_updateGradInput",
        cname("TemporalSubSampling_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},})
     wrap("TemporalSubSampling_accGradParameters",
        cname("TemporalSubSampling_accGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name=accreal},})
     wrap("TemporalRowConvolution_updateOutput",
        cname("TemporalRowConvolution_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},
        {name="int"},
        {name="int"},
        {name="boolean"},})
     wrap("TemporalRowConvolution_updateGradInput",
        cname("TemporalRowConvolution_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},
        {name="int"},
        {name="int"},
        {name="boolean"},})
     wrap("TemporalRowConvolution_accGradParameters",
        cname("TemporalRowConvolution_accGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},
        {name="int"},
        {name="int"},
        {name="boolean"},
        {name=accreal},})
     wrap("BatchNormalization_updateOutput",
        cname("BatchNormalization_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor, optional=true},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="boolean"},
        {name="double"},
        {name="double"},})
     wrap("BatchNormalization_backward",
        cname("BatchNormalization_backward"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor, optional=true},
        {name=Tensor, optional=true},
        {name=Tensor, optional=true},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="boolean"},
        {name="double"},
        {name="double"},})
     wrap("SpatialConvolutionMap_updateOutput",
        cname("SpatialConvolutionMap_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},
        {name="int"},
        {name="int"},        {name="int"},})
     wrap("SpatialConvolutionMap_updateGradInput",
        cname("SpatialConvolutionMap_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},
        {name="int"},
        {name="int"},        {name="int"},})
     wrap("SpatialConvolutionMap_accGradParameters",
        cname("SpatialConvolutionMap_accGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},
        {name="int"},
        {name="int"},        {name="int"},
        {name=accreal},})
     wrap("SpatialConvolutionMM_updateOutput",
        cname("SpatialConvolutionMM_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},})
     wrap("SpatialConvolutionMM_updateGradInput",
        cname("SpatialConvolutionMM_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},})
     wrap("SpatialConvolutionMM_accGradParameters",
        cname("SpatialConvolutionMM_accGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name=accreal},})
     wrap("SpatialDepthWiseConvolution_updateOutput",
        cname("SpatialDepthWiseConvolution_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},})
     wrap("SpatialDepthWiseConvolution_updateGradInput",
        cname("SpatialDepthWiseConvolution_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},})
     wrap("SpatialDepthWiseConvolution_accGradParameters",
        cname("SpatialDepthWiseConvolution_accGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name=accreal},})
     wrap("SpatialConvolutionLocal_updateOutput",
        cname("SpatialConvolutionLocal_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int64_t"},        {name="int64_t"},
        {name="int64_t"},        {name="int64_t"},})
     wrap("SpatialConvolutionLocal_updateGradInput",
        cname("SpatialConvolutionLocal_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int64_t"},        {name="int64_t"},
        {name="int64_t"},        {name="int64_t"},})
     wrap("SpatialConvolutionLocal_accGradParameters",
        cname("SpatialConvolutionLocal_accGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int64_t"},        {name="int64_t"},
        {name="int64_t"},        {name="int64_t"},
        {name=accreal},})
     wrap("SpatialAdaptiveMaxPooling_updateOutput",
        cname("SpatialAdaptiveMaxPooling_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name="int"},        {name="int"},})
     wrap("SpatialAdaptiveMaxPooling_updateGradInput",
        cname("SpatialAdaptiveMaxPooling_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},})
     wrap("SpatialAdaptiveAveragePooling_updateOutput",
        cname("SpatialAdaptiveAveragePooling_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},})
     wrap("SpatialAdaptiveAveragePooling_updateGradInput",
        cname("SpatialAdaptiveAveragePooling_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},})
     wrap("SpatialAveragePooling_updateOutput",
        cname("SpatialAveragePooling_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="boolean"},
        {name="boolean"},})
     wrap("SpatialAveragePooling_updateGradInput",
        cname("SpatialAveragePooling_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="boolean"},
        {name="boolean"},})
     wrap("SpatialFractionalMaxPooling_updateOutput",
        cname("SpatialFractionalMaxPooling_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="IndexTensor", noreadadd=true},
        {name=Tensor},})
     wrap("SpatialFractionalMaxPooling_updateGradInput",
        cname("SpatialFractionalMaxPooling_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="IndexTensor", noreadadd=true},})
     wrap("SpatialFullConvolution_updateOutput",
        cname("SpatialFullConvolution_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},})
     wrap("SpatialFullConvolution_updateGradInput",
        cname("SpatialFullConvolution_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},})
     wrap("SpatialFullConvolution_accGradParameters",
        cname("SpatialFullConvolution_accGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name=accreal},})
     wrap("SpatialFullConvolutionMap_updateOutput",
        cname("SpatialFullConvolutionMap_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},
        {name="int"},
        {name="int"},        {name="int"},})
     wrap("SpatialFullConvolutionMap_updateGradInput",
        cname("SpatialFullConvolutionMap_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},
        {name="int"},
        {name="int"},        {name="int"},})
     wrap("SpatialFullConvolutionMap_accGradParameters",
        cname("SpatialFullConvolutionMap_accGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},
        {name="int"},
        {name="int"},        {name="int"},
        {name=accreal},})
     wrap("SpatialDilatedConvolution_updateOutput",
        cname("SpatialDilatedConvolution_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},})
     wrap("SpatialDilatedConvolution_updateGradInput",
        cname("SpatialDilatedConvolution_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},})
     wrap("SpatialDilatedConvolution_accGradParameters",
        cname("SpatialDilatedConvolution_accGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name=accreal},})
     wrap("SpatialFullDilatedConvolution_updateOutput",
        cname("SpatialFullDilatedConvolution_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},})
     wrap("SpatialFullDilatedConvolution_updateGradInput",
        cname("SpatialFullDilatedConvolution_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},})
     wrap("SpatialFullDilatedConvolution_accGradParameters",
        cname("SpatialFullDilatedConvolution_accGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name=accreal},})
     wrap("SpatialMaxPooling_updateOutput",
        cname("SpatialMaxPooling_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="boolean"},})
     wrap("SpatialMaxPooling_updateGradInput",
        cname("SpatialMaxPooling_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="boolean"},})
     wrap("SpatialDilatedMaxPooling_updateOutput",
        cname("SpatialDilatedMaxPooling_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="boolean"},})
     wrap("SpatialDilatedMaxPooling_updateGradInput",
        cname("SpatialDilatedMaxPooling_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="boolean"},})
     wrap("SpatialMaxUnpooling_updateOutput",
        cname("SpatialMaxUnpooling_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name="int"},        {name="int"},})
     wrap("SpatialMaxUnpooling_updateGradInput",
        cname("SpatialMaxUnpooling_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name="int"},        {name="int"},})
     wrap("SpatialSubSampling_updateOutput",
        cname("SpatialSubSampling_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},})
     wrap("SpatialSubSampling_updateGradInput",
        cname("SpatialSubSampling_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},})
     wrap("SpatialSubSampling_accGradParameters",
        cname("SpatialSubSampling_accGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name=accreal},})
     wrap("SpatialUpSamplingNearest_updateOutput",
        cname("SpatialUpSamplingNearest_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name="int"},})
     wrap("SpatialUpSamplingNearest_updateGradInput",
        cname("SpatialUpSamplingNearest_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},})
     wrap("SpatialUpSamplingBilinear_updateOutput",
        cname("SpatialUpSamplingBilinear_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name="int"},
        {name="int"},})
     wrap("SpatialUpSamplingBilinear_updateGradInput",
        cname("SpatialUpSamplingBilinear_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name="int"},
        {name="int"},
        {name="int"},
        {name="int"},
        {name="int"},
        {name="int"},})
     wrap("SpatialGridSamplerBilinear_updateOutput",
        cname("SpatialGridSamplerBilinear_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},})
     wrap("SpatialGridSamplerBilinear_updateGradInput",
        cname("SpatialGridSamplerBilinear_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},        {name=Tensor},
        {name=Tensor},        {name=Tensor},
        {name=Tensor},})
     wrap("unfolded_acc",
        cname("unfolded_acc"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},})
     wrap("unfolded_copy",
        cname("unfolded_copy"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},})
     wrap("VolumetricAveragePooling_updateOutput",
        cname("VolumetricAveragePooling_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="boolean"},        {name="boolean"},})
     wrap("VolumetricAveragePooling_updateGradInput",
        cname("VolumetricAveragePooling_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="boolean"},        {name="boolean"},})
     wrap("VolumetricConvolution_updateOutput",
        cname("VolumetricConvolution_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},})
     wrap("VolumetricConvolution_updateGradInput",
        cname("VolumetricConvolution_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},})
     wrap("VolumetricConvolution_accGradParameters",
        cname("VolumetricConvolution_accGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name=accreal},})
     wrap("VolumetricConvolutionMM_updateOutput",
        cname("VolumetricConvolutionMM_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},})
     wrap("VolumetricConvolutionMM_updateGradInput",
        cname("VolumetricConvolutionMM_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},})
     wrap("VolumetricConvolutionMM_accGradParameters",
        cname("VolumetricConvolutionMM_accGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name=accreal},})
     wrap("VolumetricFractionalMaxPooling_updateOutput",
        cname("VolumetricFractionalMaxPooling_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="IndexTensor", noreadadd=true},
        {name=Tensor},})
     wrap("VolumetricFractionalMaxPooling_updateGradInput",
        cname("VolumetricFractionalMaxPooling_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="IndexTensor", noreadadd=true},})
     wrap("VolumetricFullConvolution_updateOutput",
        cname("VolumetricFullConvolution_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},})
     wrap("VolumetricFullConvolution_updateGradInput",
        cname("VolumetricFullConvolution_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},})
     wrap("VolumetricFullConvolution_accGradParameters",
        cname("VolumetricFullConvolution_accGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name=accreal},})
     wrap("VolumetricDilatedConvolution_updateOutput",
        cname("VolumetricDilatedConvolution_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},})
     wrap("VolumetricDilatedConvolution_updateGradInput",
        cname("VolumetricDilatedConvolution_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},})
     wrap("VolumetricDilatedConvolution_accGradParameters",
        cname("VolumetricDilatedConvolution_accGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name=accreal},})
     wrap("VolumetricFullDilatedConvolution_updateOutput",
        cname("VolumetricFullDilatedConvolution_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},})
     wrap("VolumetricFullDilatedConvolution_updateGradInput",
        cname("VolumetricFullDilatedConvolution_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},})
     wrap("VolumetricFullDilatedConvolution_accGradParameters",
        cname("VolumetricFullDilatedConvolution_accGradParameters"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor, optional=true},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name=accreal},})
     wrap("VolumetricMaxPooling_updateOutput",
        cname("VolumetricMaxPooling_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="boolean"},})
     wrap("VolumetricMaxPooling_updateGradInput",
        cname("VolumetricMaxPooling_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="boolean"},})
     wrap("VolumetricDilatedMaxPooling_updateOutput",
        cname("VolumetricDilatedMaxPooling_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="boolean"},})
     wrap("VolumetricDilatedMaxPooling_updateGradInput",
        cname("VolumetricDilatedMaxPooling_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="boolean"},})
     wrap("VolumetricMaxUnpooling_updateOutput",
        cname("VolumetricMaxUnpooling_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},})
     wrap("VolumetricMaxUnpooling_updateGradInput",
        cname("VolumetricMaxUnpooling_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="IndexTensor", noreadadd=true},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},
        {name="int"},        {name="int"},        {name="int"},})
     wrap("SpatialReflectionPadding_updateOutput",
        cname("SpatialReflectionPadding_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},})
     wrap("SpatialReflectionPadding_updateGradInput",
        cname("SpatialReflectionPadding_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},})
     wrap("SpatialReplicationPadding_updateOutput",
        cname("SpatialReplicationPadding_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},})
     wrap("SpatialReplicationPadding_updateGradInput",
        cname("SpatialReplicationPadding_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},})
     wrap("FeatureLPPooling_updateOutput",
        cname("FeatureLPPooling_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},
        {name="int"},
        {name="int"},
        {name="boolean"},})
     wrap("FeatureLPPooling_updateGradInput",
        cname("FeatureLPPooling_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name=accreal},
        {name="int"},
        {name="int"},
        {name="boolean"},})
     wrap("VolumetricReplicationPadding_updateOutput",
        cname("VolumetricReplicationPadding_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},})
     wrap("VolumetricReplicationPadding_updateGradInput",
        cname("VolumetricReplicationPadding_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},
        {name="int"},        {name="int"},})
     wrap("VolumetricUpSamplingNearest_updateOutput",
        cname("VolumetricUpSamplingNearest_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name="int"},})
     wrap("VolumetricUpSamplingNearest_updateGradInput",
        cname("VolumetricUpSamplingNearest_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name=Tensor},
        {name="int"},})
     wrap("VolumetricUpSamplingTrilinear_updateOutput",
        cname("VolumetricUpSamplingTrilinear_updateOutput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name="int"},
        {name="int"},
        {name="int"},})
     wrap("VolumetricUpSamplingTrilinear_updateGradInput",
        cname("VolumetricUpSamplingTrilinear_updateGradInput"),
        {{name=THNNState, invisible=true, default="NULL"},
        {name=Tensor},
        {name=Tensor},
        {name="int"},
        {name="int"},
        {name="int"},
        {name="int"},
        {name="int"},
        {name="int"},
        {name="int"},
        {name="int"},})

   end
   interface:register(string.format("torch_%sTHNN__", Tensor))

   interface:print(string.gsub([[
static void torch_TensorTHNN_init(lua_State *L)
{
  if (!luaT_pushmetatable(L, "torch.Tensor"))
    return;

  /* register functions into the "THNN" field of the tensor metaclass */
  lua_pushstring(L, "THNN");
  lua_newtable(L);
  luaT_setfuncs(L, torch_TensorTHNN__, 0);
  lua_rawset(L, -3);
  lua_pop(L, 1);

}
]], 'Tensor', Tensor))
end

interface:print([[
int luaopen_THNNx(lua_State *L) {
  torch_FloatTensorTHNN_init(L);
  torch_DoubleTensorTHNN_init(L);
  lua_createtable(L, 0, 0);
  return 1;
}
]])

if arg[1] then
   interface:tofile(arg[1])
else
   print(interface:tostring())
end