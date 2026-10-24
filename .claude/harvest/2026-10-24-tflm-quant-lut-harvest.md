# TFLM quantization + nonlinear-LUT harvest (2026-10-24)

**Commit read:** `8f6596d` ("Port signal custom ops to TensorFlow and TFLite C APIs
(#3880)"). The fork's only branch `main` equals upstream `tensorflow/tflite-micro`
`main` at this commit, so every reference below holds for both.
All paths are relative to `tensorflow/lite/micro/kernels/`.

**Method.** Three read-only mapper agents read the units in full and wrote fact
maps. Every claim below was then re-read in source by the author of this note;
one map carried errata (a wrong file count, a misattributed name, a garbled
sentence) that are not repeated here. Facts are marked **SOURCE**; consequences
for the consuming design are marked **IMPLICATION** and are not claims about TFLM.

**Purpose.** Reference design for three 8-bit code spaces planned in
`AdaWorldAPI/ndarray` (I8Asym −128..127, I8SymNaN −127..127 with −128 = NaN,
U8 deferred) and for 256-entry code tables that keep tanh-domain values in code
space (the lance-graph `MoorePalettePairs` / Palette256 law precedent).

---

## 1. Quantize / dequantize / requantize

**SOURCE**
- Rounding is `TfLiteRound` = `std::round` (ties away from zero):
  `DECLARE_STD_GLOBAL_SWITCH1(TfLiteRound, round)` — `internal/cppmath.h:36`.
- Affine quantize, `internal/reference/quantize.h:41-50`:
  `TfLiteRound(val / scale) + zero_point`, then
  `std::min(std::max(unclamped, min_val), max_val)` with
  `min_val/max_val = std::numeric_limits<OutputT>::min()/max()`.
  So int8 saturates to the full −128..127 range; there is no −127 option and no
  NaN code. A NaN input reaches `static_cast<int32_t>(round(NaN))`, which C++
  leaves undefined; no NaN guard exists in the path.
- The same rule per channel: `quantize.h:69-82`.
- Requantize int8↔uint8 with equal scale and zero points differing by 128 is a
  bit flip: `output_data[i] = input_data[i] ^ 0x80;` — `internal/reference/requantize.h:47`
  (guarded by `same_scale` at `:33` and `zero_point_diff == ∓128` at `:44-45`).

**IMPLICATION**
- I8Asym ("as is") matches TFLM's range and saturation, and also its rounding
  (ties away). It does NOT match ONNX `QuantizeLinear`, which rounds ties to even.
  Rounding has to be an explicit field of a code space, not an assumption.
- "U8 with zero point 128 = I8Asym XOR 0x80" is not a derivation only: TFLM
  ships it as the requantize fast path.
- I8SymNaN has no precedent here: TFLM never reserves an int8 code.

## 2. Tanh kernels

### 2a. int8 tanh — fixed point, not a table

**SOURCE** `internal/reference/integer_ops/tanh.h:28-60`, prepared in `tanh.cc:61-72`.
- Prepare: `input_real_multiplier = scale · 2^(31−4)` (`kInputIntegerBits = 4`),
  `frexp` into `input_multiplier`/`input_left_shift`, and
  `input_range_radius = CalculateInputRadius(4, shift, 31)`.
- Eval: `input = x − zero_point`; `input <= −radius → kMinInt8 (−128)`;
  `input >= radius → kMaxInt8 (127)`; otherwise
  `MultiplyByQuantizedMultiplier` into Q4, gemmlowp-style fixed-point `tanh`,
  `RoundingDivideByPOT(…, 31 − kOutputScale)` with `kOutputScale = 7`
  (the kernel's output is Q0.7, step 1/128), clamp to [−128, 127].

**IMPLICATION**
- The int8 tanh output saturates to **−128 at the negative end and 127 at the
  positive end**, so it is not
  odd: read as Q0.7, tanh(−big) = −1.0 exactly and tanh(+big) = 127/128. This is a measured instance of the asymmetry H-SYM targets.
- It is also a direct conflict for I8SymNaN: an I8Asym tanh result fed into an
  I8SymNaN lane would read −128 as NaN. The two code spaces must never share an
  output buffer; a code law that targets I8SymNaN must clamp to −127.

### 2b. int16 tanh — one shared, fixed 256-entry table

**SOURCE** `internal/reference/integer_ops/tanh.h:63-110`, table at
`internal/common.h:705-735`, prepare at `tanh.cc:74-124`.
- Comment: "We use the LUT for sigmoid and take into account, that
  tanh(x) = 2*sigmoid(2*x) - 1" and "We scale by 3/4 to expand range
  [-8,8]->[-10.7,10.7]".
- Table: `static const uint16_t sigmoid_table_uint16[256]`, "Table of
  sigmoid(i/24) at 0.16 format"; "Both functions are symmetric, so the LUT table
  is only needed for the absolute value of the input" (`common.h:705-711`).
- Eval: input rescaled by `input_multiplier`/`input_left_shift` (all per-tensor
  scaling happens here, before the lookup); `uh = |x| >> 8` indexes the table,
  `ut = |x| & 0xFF` interpolates: `result = (ua << 8) + ut * (ub - ua)`;
  `uh >= 255` saturates to `0xFFFF << 8`.
- Sign is restored after the lookup with asymmetric rounding:
  `(input_data >= 0) ? (result - (1 << 23) + (1 << 7))
                     : (-result + (1 << 23) + (1 << 7) - 1)`, then `>>= 8`.
- Prepare requires `zero_point == 0` on input and output and a power-of-two
  output scale of 2^−15 ("fixed-point arithmetic, which intrinsically wants
  symmetric ranges (zero_point==0) and power-of-two scales"). A non-POT input
  scale is folded into a multiplier "to change input scale to 1/(3*4096)".

**IMPLICATION**
- This is design A for our substrate: **one fixed-domain tanh table shared by
  every tensor**, with an integer rescale in front of it. Our per-calibration
  table (design B: calibration and nonlinearity fused at build time) is the
  other point; the two should be measured head to head on error, cache footprint,
  NaN propagation, and integer-only execution.
- High-byte index plus low-byte interpolation is a concrete shape for a
  precision step above a plain 256-entry lookup.
- The `+ (1 << 7)` vs `+ (1 << 7) − 1` branches mean the int16 kernel is not
  exactly odd either. An odd-mirror table (half the entries plus a sign) must be
  proven bit-exact against the full table, not assumed from tanh's symmetry.

### 2c. uint8 / float tanh

**SOURCE** `internal/reference/tanh.h`: the uint8 overload (`:82-118`) clamps to
`0` (`:98`) and `255` (`:100`) at the radius and otherwise reinterprets the result
as Q0.7 in uint8 (`:115-118`). The float path calls `std::tanh` element-wise.

## 3. The generic table builder and lookups (`internal/common.h`)

**SOURCE**
- `detail::LUTPopulateInt8` (`:521-540`): for every code `val` in the type's range,
  `dequantized = input_scale * (val - input_zero_point)`, apply `transform`
  (a `float(*)(float)` or `float(*)(float, const void*)` with type-erased params,
  `:495-513`), `TfLiteRound(transformed / output_scale) + output_zero_point`,
  clamp to the type's range, store at `lut[(uint8_t)(T)val]`.
  Comment (`:516-519`): the int8 table is "ordered as [0, 1, ..., 127, -128, ...,
  -2, -1]" because indices are cast straight to uint8.
- `detail::LUTPopulateInt16` (`:547-601`): 512 steps across the input range plus
  one end entry (513 total); each entry is the rounded sample minus half the
  midpoint interpolation error (`bias = TfLiteRound(midpoint_err / 2)`), clamped.
- `LUTSize<T>()` (`:660`): 256 for 8-bit, 513 for int16 ("the high 9 bits are used
  for indexing and the 7 remaining bits are used for interpolation").
- `LUTLookup` int16 (`:673-692`): `index = 256 + (value >> 7)`, slope from
  `lut[index+1] − lut[index]`, `delta = (slope * offset + 64) >> 7`.
- `LUTLookup` int8 (`:695`): `lut[static_cast<uint8_t>(value)]`; uint8 (`:701`):
  `lut[value]` — "int8_t -> int8_t table lookup without interpolation".
- In this commit no TFLM tanh kernel calls `LUTPopulate`; among the `.cc`
  kernels only `softmax_common.cc` does. The 256-entry int8 tanh table built by
  `LUTPopulate` is a desktop TFLite (`tensorflow/lite/kernels`) design, not a
  Micro one.

**IMPLICATION**
- `LUTPopulateInt8` is exactly our "bake once, decode → f → encode" step: the
  float work lives entirely in preparation, the hot path is one byte load
  returning a code. That is the unary form of the Palette256 law.
- The table key is (input scale, input zero point, output scale, output zero
  point, transform, params): calibration, not mode. Our `CodeLut256` key must
  carry the full decode affine plus the function identity for the same reason.
- Index order `(uint8)(int8)val` is the natural convention for a 256-entry
  i8-keyed table; an off-by-128 indexing bug is the failure class to test for.
- The 513-entry int16 builder pre-compensates interpolation error at build time.
  A higher-precision code table should do the same rather than interpolate naively.

## 4. `DecodeStateLut` — code-to-value decode of compressed weights

**SOURCE** `decode_state_lut.h:26-90`: indices of 1–7 bits
(`kMaxBitWidth = 7`, `compressed_bit_width_` "1 to 7 bits") index a per-channel
value table (`value_table_`, `value_table_channel_stride_` ≤ 128) carried in the
model flatbuffer; specialised decoders exist for widths 4/16, 3/32 and 2/16 int8,
plus a generic `DecompressToBufferWidthAny<T>`.

**IMPLICATION** Precedent for a codebook decode where the table is model data and
the index width is below 8 bits. It materialises the decoded values into a
buffer; our design keeps the values as codes and decodes only at a boundary.

## 5. What we take, and what we do not

| TFLM mechanism | taken as | note |
|---|---|---|
| full-range saturating int8, ties away | I8Asym "as is" | matches today's ndarray quantizer |
| no reserved int8 code | — | I8SymNaN is new; no TFLM precedent |
| `^ 0x80` int8↔uint8 requantize | U8 offset = I8Asym ^ 0x80 | deferred U8 mode |
| `LUTPopulateInt8` + `LUTLookup` | unary code law, baked once | design B |
| int16 shared `sigmoid_table_uint16` + rescale | fixed-domain shared table | design A, to be measured against B |
| 513-entry builder with midpoint bias | higher-precision variant | later |
| int8 tanh saturating to −128 | negative example | asymmetric output; never mix with I8SymNaN |
| `DecodeStateLut` | codebook-decode precedent | we stay in code space instead |

Not covered here: per-target kernels (`cmsis_nn/`, `xtensa/`, `ceva/`), the
gemmlowp `fixedpoint.h` tanh body, and logistic/hard_swish beyond the shared
table. NaN behaviour was read from the code path only, not run.
