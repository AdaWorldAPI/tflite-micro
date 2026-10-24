# .claude/ — AdaWorldAPI research notes

This folder holds AdaWorldAPI research notes and harvests made over this fork.
It changes no TensorFlow Lite Micro source, build or test. Upstream code is
read here as a reference design for quantized nonlinear operators in the
`AdaWorldAPI/ndarray` and `AdaWorldAPI/lance-graph` Rust crates.

- `harvest/` — dated harvests: what the upstream code does, with file:line
  references against the commit named in each file, and what it implies for
  the consuming design. Facts are quoted from source; implications are labelled.
